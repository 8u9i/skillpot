# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in the .axon format, its reference implementation, or any of its tooling, please report it privately.

**Do not file a public issue.** Instead, send a description to the project maintainers via a private security advisory on GitHub:

1. Go to https://github.com/8u9i/axon/security/advisories
2. Click "Report a vulnerability"
3. Provide:
   - A description of the vulnerability
   - Steps to reproduce it
   - The potential impact
   - Any suggested fix (if applicable)

You should receive a response within 48 hours. If you don't, follow up via the same channel.

## Scope

The following are in scope:

- The .axon binary format specification and parsers
- The Rust core library (`axon-core`)
- The C FFI bindings (`axon-ffi`)
- The CLI tool (`axon-cli`)
- The Python bindings (`axon` package)

## Out of Scope

- Third-party code or tools that consume the .axon format
- Issues that require physical access to a machine
- Issues in dependencies that are already fixed upstream

## Preferred Encryption

GitHub security advisories support encrypted communication. If you must communicate outside of GitHub, contact the maintainers directly.

## Recognition

We maintain a hall of fame for researchers who report valid security issues. With your permission, we'll acknowledge your contribution in release notes and project documentation.
