// build.rs
use std::process::Command;

fn main() {
    let hash = Command::new("git")
        .args(["rev-parse", "--short=8", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout).ok()
            } else {
                None
            }
        })
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let dirty = Command::new("git")
        .args(["diff", "--quiet"])
        .output()
        .map(|o| !o.status.success())
        .unwrap_or(false)
        || Command::new("git")
            .args(["diff", "--cached", "--quiet"])
            .output()
            .map(|o| !o.status.success())
            .unwrap_or(false);

    let suffix = if dirty { "-dirty" } else { "" };
    println!("cargo:rustc-env=GIT_COMMIT_HASH={}{}", hash, suffix);
    if let Ok(o) = Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        && o.status.success()
    {
        let stdout = String::from_utf8(o.stdout).unwrap();
        let branch = stdout.trim();
        if branch != "HEAD" {
            println!("cargo:rerun-if-changed=.git/refs/heads/{}", branch);
        }
    }
    println!("cargo:rerun-if-changed=.git/HEAD");
}
