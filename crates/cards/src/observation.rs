use super::*;
use rbp_core::Arbitrary;
use rbp_core::RiverFeatureSpec;
use rbp_core::Probability;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use std::cmp::Ordering;
use std::hash::DefaultHasher;
use std::hash::Hash;
use std::hash::Hasher;

/// A player's view of the game: hole cards plus visible board.
///
/// Observations are the atomic units of poker abstraction. Each observation
/// encodes all card information available to a player at a given point,
/// ignoring action history (which is tracked separately in the game tree).
///
/// # Operations
///
/// - [`Observation::children`] — Iterate over all possible next-street continuations
/// - [`Observation::river_scalar`] — Compute multiplayer river scalar for clustering
/// - [`Observation::equity`] — Default shorthand using the workspace clustering config
/// - [`Observation::street`] — Infer the current street from card counts
///
/// # Serialization
///
/// Observations serialize to `i64` by packing cards into bytes, enabling
/// efficient database storage. The separator `~` distinguishes hole from board
/// in string representation.
#[derive(Copy, Clone, Hash, Eq, PartialEq, Debug, PartialOrd, Ord)]
pub struct Observation {
    pocket: Hand,
    public: Hand,
}

impl Observation {
    /// Iterates over all possible next-street observations.
    ///
    /// Each child represents dealing the appropriate number of new cards
    /// (3 for flop, 1 for turn/river) from the remaining deck.
    pub fn children<'a>(&'a self) -> impl Iterator<Item = Self> + 'a {
        let n = self.street().next().n_revealed();
        HandIterator::from((n, Hand::from(*self)))
            .map(|reveal| Hand::add(self.public, reveal))
            .map(|public| Self::from((self.pocket, public)))
    }
    /// Computes the multiplayer river scalar used by clustering.
    ///
    /// The current scalar is expected pot share against opponents sampled
    /// uniformly from the remaining deck. Heads-up is computed exactly; larger
    /// tables use deterministic Monte Carlo sampling for tractability.
    pub fn river_scalar(&self, spec: &RiverFeatureSpec) -> Probability {
        debug_assert!(self.street() == Street::Rive);
        spec.validate();
        match spec.villains() {
            1 => self.exact_river_share(),
            _ => self.sampled_river_share(spec),
        }
    }

    /// Default shorthand for the workspace's multiplayer clustering scalar.
    ///
    /// This is intentionally routed through an explicit runtime spec so call
    /// sites no longer bake in heads-up assumptions by accident.
    pub fn equity(&self) -> Probability {
        self.river_scalar(&RiverFeatureSpec::default())
    }
    /// Monte Carlo equity estimation (not yet implemented).
    pub fn simulate(&self, _: usize) -> Probability {
        todo!("run out some number of simulations and take equity as average")
    }
    /// Infers the street from total observed cards.
    pub fn street(&self) -> Street {
        Street::from(self.public().size() + self.pocket().size())
    }
    /// The player's hole cards.
    pub fn pocket(&self) -> &Hand {
        &self.pocket
    }
    /// The community board cards.
    pub fn public(&self) -> &Hand {
        &self.public
    }
    /// Iterates over all possible opponent observations.
    ///
    /// Returns observations with the same board but different hole cards,
    /// excluding any cards already visible to hero. For river, this yields
    /// C(45, 2) = 990 possible opponent holdings.
    pub fn opponents(&self) -> impl Iterator<Item = Self> + '_ {
        HandIterator::from((2, Hand::from(*self)))
            .map(|hole| (hole, self.public))
            .map(Self::from)
    }
    /// String separator between hole and board in display format.
    pub const SEPARATOR: &'static str = "~";

    fn exact_river_share(&self) -> Probability {
        let hero = Strength::from(Hand::from(*self));
        let (share, total) = self
            .opponents()
            .map(Hand::from)
            .map(Strength::from)
            .fold((0.0, 0u32), |(sum, n), villain| {
                (
                    sum + Self::showdown_share(hero, std::iter::once(villain)),
                    n + 1,
                )
            });
        match total {
            0 => 0.5,
            _ => share / total as Probability,
        }
    }

    fn sampled_river_share(&self, spec: &RiverFeatureSpec) -> Probability {
        let board = self.public;
        let hero = Strength::from(Hand::from(*self));
        let needed = spec.villains() * 2;
        let deck = Vec::<Card>::from(Hand::from(*self).complement());
        debug_assert!(needed <= deck.len(), "not enough cards for multiplayer rollout");
        let mut hasher = DefaultHasher::default();
        self.hash(&mut hasher);
        spec.hash(&mut hasher);
        let mut rng = SmallRng::seed_from_u64(hasher.finish());
        let mut total = 0.0;
        let mut cards = deck.clone();
        for _ in 0..spec.samples {
            cards.clone_from(&deck);
            for i in 0..needed {
                let j = rng.random_range(i..cards.len());
                cards.swap(i, j);
            }
            let villains = cards[..needed].chunks_exact(2).map(|chunk| {
                let hole = chunk.iter().copied().collect::<Hand>();
                Strength::from(Hand::add(hole, board))
            });
            total += Self::showdown_share(hero, villains);
        }
        total / spec.samples as Probability
    }

    fn showdown_share(
        hero: Strength,
        villains: impl IntoIterator<Item = Strength>,
    ) -> Probability {
        let mut strengths = Vec::with_capacity(1);
        strengths.push(hero);
        strengths.extend(villains);
        let best = strengths
            .iter()
            .copied()
            .max()
            .expect("hero is always present");
        let winners = strengths.iter().filter(|&&strength| strength == best).count();
        match hero.cmp(&best) {
            Ordering::Equal => 1.0 / winners as Probability,
            _ => 0.0,
        }
    }
}
/// i64 isomorphism
///
/// Packs all the cards in order, starting from LSBs.
/// Good for database serialization. Interchangable with u64
impl From<Observation> for i64 {
    fn from(observation: Observation) -> Self {
        std::iter::empty::<Card>()
            .chain(observation.public.into_iter())
            .chain(observation.pocket.into_iter())
            .map(|card| 1 + u8::from(card) as u64) // distinguish 0x00 and 2c
            .fold(0u64, |acc, card| acc << 8 | card) as i64 // next card
    }
}

impl From<i64> for Observation {
    fn from(bits: i64) -> Self {
        Self::from(
            (0u64..8u64)
                .map(|i| bits >> (i * 8))
                .take_while(|&bits| bits > 0)
                .map(|bits| bits as u8)
                .map(|bits| bits - 1) // distinguish 0x00 and 2c
                .map(Card::from)
                .map(Hand::from)
                .enumerate()
                .fold(
                    (Hand::empty(), Hand::empty()),
                    |(pocket, public), (i, hand)| {
                        if i < 2 {
                            (Hand::add(pocket, hand), public)
                        } else {
                            (pocket, Hand::add(public, hand))
                        }
                    },
                ),
        )
    }
}

/// assemble Observation from private + public Hands
impl From<(Hand, Hand)> for Observation {
    fn from((pocket, public): (Hand, Hand)) -> Self {
        debug_assert!(pocket.size() == 2);
        debug_assert!(public.size() <= 5);
        Self { pocket, public }
    }
}

/// Generate a random observation for a given street
impl From<Street> for Observation {
    fn from(street: Street) -> Self {
        let mut deck = Deck::new();
        let n = street.n_observed();
        let pocket = (0..2)
            .map(|_| deck.draw())
            .map(u64::from)
            .map(Hand::from)
            .fold(Hand::empty(), Hand::add);
        let public = (2..n)
            .map(|_| deck.draw())
            .map(u64::from)
            .map(Hand::from)
            .fold(Hand::empty(), Hand::add);
        Self::from((pocket, public))
    }
}

/// what is our belief of the deck from this perspective
impl From<Observation> for Deck {
    fn from(observation: Observation) -> Self {
        Self::from(Hand::from(observation).complement())
    }
}

/// coalesce public + private cards into single Hand
impl From<Observation> for Hand {
    fn from(observation: Observation) -> Self {
        Self::add(observation.pocket, observation.public)
    }
}

impl From<(Hole, Board)> for Observation {
    fn from((hole, board): (Hole, Board)) -> Self {
        Self::from((Hand::from(hole), Hand::from(board)))
    }
}

/// losing ordering information, reduce revealed cards into Observation
impl TryFrom<Vec<Card>> for Observation {
    type Error = String;
    fn try_from(cards: Vec<Card>) -> Result<Self, Self::Error> {
        if cards
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            == cards.len()
        {
            match cards.len() {
                2 | 5 | 6 | 7 => Ok(Self::from((
                    Hand::from(cards[..2].to_vec()),
                    Hand::from(cards[2..].to_vec()),
                ))),
                _ => Err(format!("invalid card count: {}", cards.len())),
            }
        } else {
            Err(format!("duplicate cards: {}", cards.len()))
        }
    }
}

impl TryFrom<&str> for Observation {
    type Error = String;
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        let (pocket, public) = s
            .trim()
            .split_once(Self::SEPARATOR)
            .unwrap_or((s.trim(), ""));
        let pocket = Hand::try_from(pocket)?;
        let public = Hand::try_from(public)?;
        if Hand::overlaps(&pocket, &public) {
            return Err(format!("duplicate cards between pocket and board"));
        }
        match (pocket.size(), public.size()) {
            (2, 0) | (2, 3) | (2, 4) | (2, 5) => Ok(Self::from((pocket, public))),
            _ => Err(format!("invalid card counts: {} {}", pocket, public)),
        }
    }
}

impl Arbitrary for Observation {
    fn random() -> Self {
        Self::from(Street::random())
    }
}

/// display Observation as pocket + public
impl std::fmt::Display for Observation {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} {} {}", self.pocket, Self::SEPARATOR, self.public)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bijective_i64() {
        let random = Observation::random();
        assert!(random == Observation::from(i64::from(random)));
    }

    #[test]
    fn opponents_count() {
        assert_eq!(Observation::from(Street::Rive).opponents().count(), 0990); // C(45, 2)
        assert_eq!(Observation::from(Street::Turn).opponents().count(), 1035); // C(46, 2)
        assert_eq!(Observation::from(Street::Flop).opponents().count(), 1081); // C(47, 2)
        assert_eq!(Observation::from(Street::Pref).opponents().count(), 1225); // C(50, 2)
    }

    #[test]
    fn river_scalar_handles_split_pots() {
        let obs = Observation::try_from("2c 3d ~ Ah Kh Qh Jh Th").unwrap();
        let heads_up = RiverFeatureSpec::default()
            .with_players_total(2)
            .with_players_alive(2);
        let six_max = RiverFeatureSpec::default();
        assert!((obs.river_scalar(&heads_up) - 0.5).abs() < 1e-6);
        assert!((obs.river_scalar(&six_max) - (1.0 / 6.0)).abs() < 1e-6);
    }

    #[test]
    fn river_scalar_is_deterministic_for_fixed_spec() {
        let obs = Observation::try_from("Ah Kd ~ Qh Jh 2c 7d 3s").unwrap();
        let spec = RiverFeatureSpec::default().with_samples(128);
        assert_eq!(obs.river_scalar(&spec), obs.river_scalar(&spec));
    }
}
