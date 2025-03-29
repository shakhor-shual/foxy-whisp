import numpy as np

def calculate_vu_meter(audio_chunk: np.ndarray) -> int:
    """
    Calculates the signal level in the range 0-100.
    Enhanced stability and edge case handling.
    """
    if not isinstance(audio_chunk, np.ndarray) or len(audio_chunk) == 0:
        return 0
    
    # Handle NaN and Inf values
    if np.any(~np.isfinite(audio_chunk)):
        clean_chunk = audio_chunk[np.isfinite(audio_chunk)]
        if len(clean_chunk) == 0:
            return 0
        audio_chunk = clean_chunk

    # Ensure float32 type and -1 to 1 range
    if audio_chunk.dtype != np.float32:
        audio_chunk = audio_chunk.astype(np.float32)
    if np.max(np.abs(audio_chunk)) > 1.0:
        audio_chunk = audio_chunk / 32768.0

    # Calculate RMS with numeric stability
    rms = np.sqrt(np.mean(np.square(audio_chunk)) + 1e-10)
    
    # Convert to dB with safe log
    db = 20 * np.log10(max(rms, 1e-10))
    
    # Updated mapping points for smoother response
    db_points = np.array([-60, -40, -20, -10, 0])
    level_points = np.array([0, 40, 75, 90, 100])

    # Clamp values
    if db <= db_points[0]:
        return 0
    if db >= db_points[-1]:
        return 100
        
    # Find interval for interpolation
    idx = np.searchsorted(db_points, db) - 1
    idx = max(0, min(idx, len(db_points) - 2))
    
    # Linear interpolation
    db_range = db_points[idx + 1] - db_points[idx]
    level_range = level_points[idx + 1] - level_points[idx]
    fraction = (db - db_points[idx]) / db_range
    level = level_points[idx] + fraction * level_range
    
    return int(round(max(0, min(100, level))))
