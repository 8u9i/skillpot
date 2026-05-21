use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::Path;
use std::ptr;
use axon_core::*;

pub struct AxonHandle { file: MappedAxonFile<'static> }

#[no_mangle]
pub unsafe extern "C" fn axon_open(path: *const c_char) -> *mut AxonHandle {
    if path.is_null() { return ptr::null_mut(); }
    let cstr = unsafe { CStr::from_ptr(path) };
    let path_str = match cstr.to_str() { Ok(s) => s, Err(_) => return ptr::null_mut() };
    match MappedAxonFile::open(Path::new(path_str)) {
        Ok(file) => Box::into_raw(Box::new(AxonHandle { file })) as *mut AxonHandle,
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn axon_close(handle: *mut AxonHandle) {
    if !handle.is_null() { unsafe { drop(Box::from_raw(handle)); } }
}

#[no_mangle]
pub unsafe extern "C" fn axon_tensor_count(handle: *const AxonHandle) -> u64 {
    if handle.is_null() { return 0; } unsafe { (&*handle).file.file.header.tensor_count }
}

#[no_mangle]
pub unsafe extern "C" fn axon_payload_size(handle: *const AxonHandle) -> u64 {
    if handle.is_null() { return 0; } unsafe { (&*handle).file.file.header.payload_size }
}

#[no_mangle]
pub unsafe extern "C" fn axon_model_name(handle: *const AxonHandle, buf: *mut c_char, buf_size: u64) -> u64 {
    if handle.is_null() || buf.is_null() || buf_size == 0 { return 0; }
    let model = unsafe { (&*handle) }.file.file.manifest.model.as_deref().unwrap_or("");
    let cstr = CString::new(model).unwrap_or_default();
    let bytes = cstr.as_bytes_with_nul();
    let to_copy = bytes.len().min(buf_size as usize);
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, to_copy); }
    (to_copy.saturating_sub(1)) as u64
}

#[no_mangle]
pub unsafe extern "C" fn axon_tensor_info(
    handle: *const AxonHandle, index: u64,
    name_buf: *mut c_char, name_buf_size: u64,
    dtype_out: *mut u32, rank_out: *mut u32,
    shape_out: *mut u64, data_offset_out: *mut u64, data_size_out: *mut u64,
) -> i32 {
    if handle.is_null() { return 0; }
    let handle = unsafe { &*handle };
    let order = &handle.file.file.manifest.tensor_order;
    if index as usize >= order.len() { return 0; }
    let name = &order[index as usize];
    let desc = match handle.file.file.manifest.get_tensor(name) { Some(d) => d, None => return 0 };
    let cstr = CString::new(desc.name_str()).unwrap_or_default();
    let bytes = cstr.as_bytes_with_nul();
    if !name_buf.is_null() && name_buf_size > 0 {
        let to_copy = bytes.len().min(name_buf_size as usize);
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), name_buf as *mut u8, to_copy); }
    }
    if !dtype_out.is_null() { unsafe { *dtype_out = desc.dtype; } }
    if !rank_out.is_null() { unsafe { *rank_out = desc.rank; } }
    if !shape_out.is_null() { unsafe { for i in 0..desc.rank as usize { *shape_out.add(i) = desc.shape[i]; } } }
    if !data_offset_out.is_null() { unsafe { *data_offset_out = desc.data_offset; } }
    if !data_size_out.is_null() { unsafe { *data_size_out = desc.data_size; } }
    1
}

#[no_mangle]
pub unsafe extern "C" fn axon_tensor_data(handle: *const AxonHandle, index: u64, data_size: *mut u64) -> *const u8 {
    if handle.is_null() { return ptr::null(); }
    let handle = unsafe { &*handle };
    let order = &handle.file.file.manifest.tensor_order;
    if index as usize >= order.len() { return ptr::null(); }
    let name = &order[index as usize];
    let desc = match handle.file.file.manifest.get_tensor(name) { Some(d) => d, None => return ptr::null() };
    if !data_size.is_null() { unsafe { *data_size = desc.data_size; } }
    let offset = desc.data_offset as usize;
    let end = offset + desc.data_size as usize;
    if end <= handle.file.file.data.len() { handle.file.file.data[offset..end].as_ptr() } else { ptr::null() }
}

#[no_mangle]
pub unsafe extern "C" fn axon_verify_checksums(handle: *const AxonHandle, failed_indices: *mut u64, failed_count: *mut u64) -> u64 {
    if handle.is_null() { return 0; }
    let handle = unsafe { &*handle };
    let results = handle.file.file.verify_all_checksums();
    let mut valid = 0u64;
    let mut failed_list = Vec::new();
    for (i, (name, ok)) in results.iter().enumerate() {
        if *ok { valid += 1; } else {
            if let Some(pos) = handle.file.file.manifest.tensor_order.iter().position(|n| n == name) { failed_list.push(pos as u64); }
        }
    }
    if !failed_indices.is_null() && !failed_list.is_empty() { unsafe { for (i, &idx) in failed_list.iter().enumerate() { *failed_indices.add(i) = idx; } } }
    if !failed_count.is_null() { unsafe { *failed_count = failed_list.len() as u64; } }
    valid
}

#[no_mangle]
pub unsafe extern "C" fn axon_version(buf: *mut c_char, buf_size: u64) -> u64 {
    let version = CString::new(format!("Axon v{}", axon_core::AXON_VERSION)).unwrap();
    let bytes = version.as_bytes_with_nul();
    if !buf.is_null() && buf_size > 0 {
        let to_copy = bytes.len().min(buf_size as usize);
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, to_copy); }
        (to_copy - 1) as u64
    } else { 0 }
}
