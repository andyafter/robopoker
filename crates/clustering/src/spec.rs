use rbp_cards::Street;
use rbp_core::*;

/// Runtime configuration for multiplayer-aware abstraction generation.
///
/// The current implementation still uses the existing compile-time bucket
/// layout, but this spec makes player count, rollout behavior, and target
/// bucket sizes explicit at the call sites.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClusteringSpec {
    river: RiverFeatureSpec,
    pub river_buckets: usize,
    pub turn_buckets: usize,
    pub flop_buckets: usize,
    pub turn_iterations: usize,
    pub flop_iterations: usize,
}

impl ClusteringSpec {
    /// Default six-max clustering configuration.
    pub const fn six_max() -> Self {
        Self {
            river: RiverFeatureSpec::six_max(),
            river_buckets: KMEANS_EQTY_CLUSTER_COUNT,
            turn_buckets: KMEANS_TURN_CLUSTER_COUNT,
            flop_buckets: KMEANS_FLOP_CLUSTER_COUNT,
            turn_iterations: KMEANS_TURN_TRAINING_ITERATIONS,
            flop_iterations: KMEANS_FLOP_TRAINING_ITERATIONS,
        }
    }

    /// Returns the river-feature portion of the clustering spec.
    pub const fn river(&self) -> RiverFeatureSpec {
        self.river
    }

    /// Returns a copy with a different active-player count.
    pub fn with_players_alive(mut self, players_alive: usize) -> Self {
        self.river = self.river.with_players_alive(players_alive);
        self
    }

    /// Returns a copy with a different total-seat count.
    pub fn with_players_total(mut self, players_total: usize) -> Self {
        self.river = self.river.with_players_total(players_total);
        self
    }

    /// Returns a copy with a different rollout budget.
    pub fn with_river_samples(mut self, samples: usize) -> Self {
        self.river = self.river.with_samples(samples);
        self
    }

    /// Panics if the spec does not match the current compile-time layout.
    pub fn validate(&self) {
        self.river.validate();
        assert!(
            self.river_buckets <= u8::MAX as usize + 1,
            "river bucket count exceeds u8 abstraction encoding"
        );
        assert!(
            self.flop_buckets <= u8::MAX as usize + 1,
            "flop bucket count exceeds u8 abstraction encoding"
        );
        assert!(
            self.turn_buckets <= u8::MAX as usize + 1,
            "turn bucket count exceeds u8 abstraction encoding"
        );
        assert_eq!(
            self.river_buckets,
            Street::Rive.n_abstractions(),
            "current implementation expects river bucket count to match static layout"
        );
        assert_eq!(
            self.turn_buckets,
            Street::Turn.k(),
            "current implementation expects turn bucket count to match static layout"
        );
        assert_eq!(
            self.flop_buckets,
            Street::Flop.k(),
            "current implementation expects flop bucket count to match static layout"
        );
        assert_eq!(
            self.turn_iterations,
            Street::Turn.t(),
            "current implementation expects turn iterations to match static layout"
        );
        assert_eq!(
            self.flop_iterations,
            Street::Flop.t(),
            "current implementation expects flop iterations to match static layout"
        );
    }
}

impl Default for ClusteringSpec {
    fn default() -> Self {
        Self::six_max()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_spec_matches_static_layout() {
        ClusteringSpec::default().validate();
    }

    #[test]
    fn spec_supports_player_count_changes() {
        let spec = ClusteringSpec::default()
            .with_players_total(7)
            .with_players_alive(4)
            .with_river_samples(32);
        assert_eq!(spec.river().players_total, 7);
        assert_eq!(spec.river().players_alive, 4);
        assert_eq!(spec.river().samples, 32);
    }
}
