# Changelog

## 0.3.4 (unreleased)

### Fixed

- Validate streamed inference protocol records and reject malformed responses
  and ambiguous JSON before issuing benchmark evidence (#23, #25).
- Reject unrepresentable and impossible benchmark metrics (#24).
- Check installed distributions with isolated Python and verify dependency
  consistency in the canonical release gate.

Earlier releases are documented in GitHub Releases.
