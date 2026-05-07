//! Autotrain Binary
//!
//! Unified training pipeline with postgres as source of truth.
//!
//! Options: --status, --fast, --slow, --cluster, --reset

use protobuf::Message;
use std::fs::File;
use std::io::Write;

#[tokio::main]
async fn main() {
    let should_profile =
        !std::env::args().any(|arg| matches!(arg.as_str(), "--status" | "--reset"));
    let guard = if should_profile {
        Some(pprof::ProfilerGuard::new(10).unwrap())
    } else {
        None
    };

    rbp_core::log();
    rbp_core::kys();
    rbp_core::brb();
    rbp_autotrain::Mode::run().await;

    if let Some(guard) = guard
        && let Ok(report) = guard.report().build()
    {
        let mut file = File::create("profile.pb").unwrap();
        let profile = report.pprof().unwrap();

        let content = profile.write_to_bytes().unwrap();
        file.write_all(&content).unwrap();
    }
}
