use xxhash_rust::xxh3::Xxh3;

pub fn xxh3_64(data: &[u8]) -> u64 {
    let mut hasher = Xxh3::new();
    hasher.update(data);
    hasher.digest()
}

pub fn verify_checksum(data: &[u8], expected: u64) -> bool {
    xxh3_64(data) == expected
}
