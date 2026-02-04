# Tree regenerated from JSON


## Entry
- process_image (file: gpuphot/phot/photo_gpu.py:625-697)
  - type: Mixed
  - notes: Entry point: orchestra calibration and post-processing (calls `calibrate_image`).
  - calls (textual order):
    - calibrate_image (file: gpuphot/phot/photo_gpu.py:2837-3145)
      - type: Mixed
      - notes: Convert input to GPU if applicable (cp.asarray(imdata)). Background orchestration, detection, star_dataset creation, photometry and catalog queries.
      - calls (textual order):
        - get_local_background_fft (file: gpuphot/phot/background.py:25-89)
          - type: GPU
          - notes: Secure CuPy (cp.array) and calculate background per tiles; calls get_mean_std and fill_nan_fft.
          - calls (textual order):
            - get_mean_std (file: gpuphot/phot/conv.py:84-109)
              - type: GPU
              - notes: Call gen_apm_filter and convolve_fft (FFT-based).
            - fill_nan_fft (file: gpuphot/phot/conv.py)
              - type: GPU
              - notes: Fill NaNs using FFT operations; can iterate.
            - decompose_into_tiles (file: gpuphot/phot/utils.py)
              - type: CPU/GPU useful
              - notes: Helper for tiling when processing the image by sections is necessary.
        - detect_isolated_stars (file: gpuphot/phot/psf.py:123-205)
          - type: Mixed
          - notes: Detect isolated stars using convolutions on GPU; for KDTree it does transfer GPU->CPU (.get()).
          - calls (textual order):
            - detect_sources_psf (file: gpuphot/phot/psf.py:678-716)
              - type: GPU
            - get_centroids_distance_kdtree (KDTree)
              - type: CPU
              - notes: Runs on CPU; therefore `coor_f.get()` is made from GPU before this call.
        - create_star_dataset (file: gpuphot/phot/psf.py:209-321)
          - type: GPU
          - notes: Create cutouts by star; converts inputs to CuPy if they come from NumPy (cp.asarray).
        - perform_opt_photometry (file: gpuphot/phot/photo_gpu.py:701-1009)
          - type: Mixed
          - notes: Optimized photometry main flow; contains branches that use crossmatch and clustering (cuML optional).
          - calls (textual order):
            - create_aperture_corrections_map_gpu (file: gpuphot/phot/photo_gpu.py:388-565)
              - type: Mixed
              - notes: Calculate aperture curves and group stars (call `group_star_dataset`).
              - calls (textual order):
                - batch_aperture_photometry (file: gpuphot/phot/photo_gpu.py:1789-2423)
                  - type: GPU
                  - notes: Version with streams and OOM handling; accepts CuPy arrays (if input NumPy -> cp.asarray(img)).
                  - on_exception:```
                    {
  "description": "Si falla por memoria u otro error en GPU",
  "actions": [
    "liberar mempool (mempool.free_all_blocks())",
    "reintentar con ajustes internos / reducir tile-size",
    "si sigue fallando -> fallback CPU o devolver None (según implementación)"
  ]
}
                    ```- grouping_dispatch:
                - group_star_dataset  (archivo: gpuphot/phot/psf.py:471-541)
                  - tipo: Mixto
                  - decision:```
                    {
  "condition": "is_gpu_input == True",
  "then": {
    "check": "CUML_CLUSTERING_AVAILABLE == True",
    "then": {
      "call": "_group_star_dataset_gpu_impl",
      "notes": "Intento GPU (cuML). Si excepción -> fallback CPU (transfer coords.get())."
    },
    "else": {
      "call": "_group_star_dataset_cpu_impl",
      "notes": "Fallback a CPU si cuML clustering no disponible."
    }
  },
  "else": {
    "call": "_group_star_dataset_cpu_impl",
    "notes": "Input es CPU -> usar CPU grouping directamente."
  }
}
                    ```- find_aperture_corrections_gpu (file: gpuphot/phot/photo_gpu.py:301-353)
              - type: GPU
              - notes: Crossmatches GPU when possible (calls `crossmatch_sources`).
            - crossmatch_sources (file: gpuphot/utils/catalog.py:137-215)
              - type: Mixed
              - decision:```
                {
  "condition": "is_gpu_input",
  "then": {
    "check": "CUML_AVAILABLE",
    "then": {
      "call": "_crossmatch_sources_gpu_impl",
      "notes": "Usa cuML NearestNeighbors en GPU; exige CuPy arrays; si falla -> excepción capturada y fallback CPU."
    },
    "else": {
      "call": "_crossmatch_sources_cpu_impl",
      "notes": "cuML no disponible -> usar CPU KDTree (transferencias .get() si la entrada era GPU)."
    }
  },
  "else": {
    "call": "_crossmatch_sources_cpu_impl",
    "notes": "Entrada era CPU -> usar KDTree en CPU."
  },
  "on_gpu_exception": {
    "actions": [
      "logger.warning + free mempool if needed",
      "transfer: source_coords.get(), ref_coords.get()",
      "call _crossmatch_sources_cpu_impl (KDTree)",
      "if original input was GPU -> cp.asarray(result_np) to return CuPy arrays"
    ]
  }
}
                ```- batch_aperture_photometry (main call)  (archivo: gpuphot/phot/photo_gpu.py:1789-2423)
              - tipo: GPU
              - notas: Se usa para calcular fluxes y áreas; ya aparece también dentro de create_aperture_corrections_map_gpu.
        - catalog_results  (archivo: gpuphot/utils/catalog.py)
          - tipo: CPU
          - notas: Consulta Vizier y post-procesa catálogo (opera en CPU).

## Notas generadas

- **conditions_extracted**: ['CUML_AVAILABLE (gpuphot/utils/catalog.py:35-43)', 'CUML_CLUSTERING_AVAILABLE (gpuphot/phot/psf.py:28-42)', 'is_gpu_input checks (isinstance(..., cp.ndarray))']
- **transfers_examples**: ['cp.asarray(imdata) in calibrate_image (photo_gpu.py:2849)', 'coor_f.get() in detect_isolated_stars (psf.py:175)', 'source_coords.get() / cp.asarray(result_np) in crossmatch fallback (catalog.py:186,206)']
- **oom_handling**: Representado en nodes con 'on_exception' y OOM_DECISION en DOT; el comportamiento es reactivo (try/except + mempool.free_all_blocks() + retry/fallback).