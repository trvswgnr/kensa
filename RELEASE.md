# Release Instructions

1. bump version number in Cargo.toml
2. run `cargo build --release`
3. make commits for any outstanding changes and push
4. run `TAG=$(grep '^version = ' Cargo.toml | cut -d'"' -f2) git tag v$TAG && git push origin v$TAG`
5. go to GitHub repo and create new release, select the latest tag, write a nice description...
6. upload ./target/release/kensa binary to release
7. publish
