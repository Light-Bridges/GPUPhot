Citing GPUPhot
==============

If you use **GPUPhot** in academic research, observations, or scientific publications, please cite the framework paper and reference the Astrophysics Source Code Library (ASCL) record.

Companion Manuscripts
---------------------

1. **Framework & Distributed Architecture (Paper):**
   Lemes-Perera, S., Alarcon, M. R., Serra-Ricart, M., & Caballero-Gil, P.
   *"GPUPHOT: A Python Framework for High-Performance GPU-Accelerated Photometry and Distributed Astronomical Data Reduction"*,
   *Astronomy and Computing* (submitted, 2026). `arXiv:2609.32375 <https://arxiv.org/abs/2609.32375>`_ [astro-ph.IM], `doi:10.48550/arXiv.2609.32375 <https://doi.org/10.48550/arXiv.2609.32375>`_.

2. **Kernel-Based Photometric Algorithms (Companion Paper):**
   Alarcon, M. R., Lemes-Perera, S., Serra-Ricart, M., & Licandro, J.
   *"GPUPHOT: Kernel-Based Algorithms for Point-Source Detection and Photometry with a Spatially Variable PSF"*,
   *The Planetary Science Journal* (in preparation, 2026).

Persistent Software Archive (Zenodo)
------------------------------------

GPUPhot is archived in Zenodo with a permanent Digital Object Identifier:

* **DOI:** `10.5281/zenodo.23098402 <https://doi.org/10.5281/zenodo.23098402>`_
* **Release:** v1.0.1

Astrophysics Source Code Library (ASCL)
---------------------------------------

GPUPhot is registered in the `Astrophysics Source Code Library <https://ascl.net/>`_:

* **ASCL Record:** ``ascl:XXXX.XXX``
* **NASA ADS Bibcode:** ``YYYYascl.soft...S``

When citing the code package directly in papers using the standard AASTeX/ASCL convention:

.. code-block:: latex

   \software{GPUPhot \citep{gpuphot_ascl}}

BibTeX Entries
--------------

.. code-block:: bibtex

   @article{gpuphot2026,
     author        = {Lemes-Perera, Samuel and Alarcon, Miguel R. and Serra-Ricart, Miquel and Caballero-Gil, Pino},
     title         = {{GPUPHOT: A Python Framework for High-Performance GPU-Accelerated Photometry and Distributed Astronomical Data Reduction}},
     journal       = {arXiv preprint arXiv:2609.32375},
     year          = {2026},
     eprint        = {2609.32375},
     archivePrefix = {arXiv},
     primaryClass  = {astro-ph.IM},
     doi           = {10.48550/arXiv.2609.32375},
     note          = {Submitted to Astronomy and Computing}
   }

   @article{gpuphot_algorithms2026,
     author        = {Alarcon, Miguel R. and Lemes-Perera, Samuel and Serra-Ricart, Miquel and Licandro, Javier},
     title         = {{GPUPHOT: Kernel-Based Algorithms for Point-Source Detection and Photometry with a Spatially Variable PSF}},
     journal       = {The Planetary Science Journal},
     year          = {2026},
     note          = {In preparation}
   }

   @software{gpuphot_zenodo,
     author        = {Lemes-Perera, Samuel and Alarcon, Miguel R.},
     title         = {{Light-Bridges/GPUPhot: GPUPhot v1.0.1}},
     month         = oct,
     year          = {2026},
     publisher     = {Zenodo},
     version       = {v1.0.1},
     doi           = {10.5281/zenodo.23098402},
     url           = {https://doi.org/10.5281/zenodo.23098402}
   }

   @software{gpuphot_ascl,
     author        = {Lemes-Perera, Samuel and Alarcon, Miguel R. and Serra-Ricart, Miquel and Caballero-Gil, Pino and Licandro, Javier},
     title         = {{GPUPhot: A GPU-Accelerated Framework for Astronomical Photometry and Astrometry}},
     howpublished  = {Astrophysics Source Code Library},
     year          = {2026},
     note          = {ascl:XXXX.XXX}
   }

Acknowledgements
----------------

When acknowledging GPUPhot in observational papers, please consider the following text:

.. note::

   *"This work made use of GPUPhot, a GPU-accelerated astronomical photometry and astrometry pipeline developed by Light Bridges S.L. and Universidad de La Laguna."*
