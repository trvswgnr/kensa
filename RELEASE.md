# Release Instructions

1. bump version number in Cargo.toml
1. make commits for any outstanding changes and push
1. run `TAG=$(grep '^version = ' Cargo.toml | cut -d'"' -f2) git tag v$TAG && git push origin v$TAG`
1. run `cargo build --release`
1. go to GitHub repo and create new release, select the latest tag, write a nice description...
1. upload ./target/release/kensa binary to release
1. publish