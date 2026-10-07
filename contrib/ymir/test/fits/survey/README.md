# Real survey files

Small cuts of public archive files, for the laws ymir.fits states over
bytes it did not write. `gen/survey.py` downloads each original by range
requests, checks the SHA-256 of the bytes it fetched, and cuts it. A cut
keeps every header record as the archive wrote it, except the rewritten
ones listed below (astropy formats them, comments kept), and keeps the data
bytes as stored. `values` holds astropy's reading of every image and
numeric binary-table column, as MD5 digests of float64.

## jwst-nircam-i2d.fits

- Source: <https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/jw02736-o001_t001_nircam_clear-f090w_i2d.fits>
- Archive: MAST (JWST ERO 2736, SMACS 0723, NIRCam F090W mosaic)
- Terms: Public JWST data from MAST; NASA/ESA/CSA, acknowledgement requested.
- SHA-256 of the fetched bytes: `48aa8ef717b677574aaf7a9c307b72574b97b6eea5c6c36b4445602a45d15c9b`
- Covers: the JWST pipeline's headers, a 3-axis int32 context image, and a 265-column header table written through astropy by stdatamodels.
- Cut:
  - HDU 0 (original HDU 0): as stored.
  - HDU 1 (original HDU 1): rows 2360-2391 and columns 2500-2563 of each plane (0-based); rewritten: CRPIX1 CRPIX2 NAXIS1 NAXIS2.
  - HDU 2 (original HDU 2): rows 2360-2391 and columns 2500-2563 of each plane (0-based); rewritten: NAXIS1 NAXIS2.
  - HDU 3 (original HDU 3): rows 2360-2391 and columns 2500-2563 of each plane (0-based); rewritten: NAXIS1 NAXIS2.
  - HDU 4 (original HDU 8): rows 0-3 of 72 (0-based); rewritten: NAXIS2.

## hst-acs-drz.fits

- Source: <https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HST/product/j8pu0y010_drz.fits>
- Archive: MAST (HST ACS/WFC drizzled association j8pu0y010)
- Terms: Public HST data from MAST; NASA/ESA, acknowledgement requested.
- SHA-256 of the fetched bytes: `1da722db791a26d3f8f5506fd202a083f9a921cb53df75cfac334b2687c04ca0`
- Covers: an 829-record primary header from CALACS and drizzlepac, the drizzle weight and context images, and a 293-column header table.
- Cut:
  - HDU 0 (original HDU 0): as stored.
  - HDU 1 (original HDU 1): rows 2200-2231 and columns 2100-2163 of each plane (0-based); rewritten: CRPIX1 CRPIX2 NAXIS1 NAXIS2.
  - HDU 2 (original HDU 2): rows 2200-2231 and columns 2100-2163 of each plane (0-based); rewritten: CRPIX1 CRPIX2 NAXIS1 NAXIS2.
  - HDU 3 (original HDU 3): rows 2200-2231 and columns 2100-2163 of each plane (0-based); rewritten: CRPIX1 CRPIX2 NAXIS1 NAXIS2.
  - HDU 4 (original HDU 4): as stored.

## hst-wfpc2-c0f.fits

- Source: <https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HST/product/u2ou0101t_c0f.fits>
- Archive: MAST (HST WFPC2 calibrated exposure u2ou0101t, waivered FITS)
- Terms: Public HST data from MAST; NASA/ESA, acknowledgement requested.
- SHA-256 of the fetched bytes: `29da403e340d1eab99d8bb954c86eb92d07eca1dae8e4c609d66311552137f68`
- Covers: a 3-axis float primary with BSCALE and BZERO and an ASCII table of group parameters, as STSDAS wrote them.
- Cut:
  - HDU 0 (original HDU 0): rows 380-411 and columns 380-443 of each plane (0-based); rewritten: CRPIX1 CRPIX2 NAXIS1 NAXIS2.
  - HDU 1 (original HDU 1): as stored.

## sdss-spec-lite.fits

- Source: <https://data.sdss.org/sas/dr17/sdss/spectro/redux/26/spectra/lite/0266/spec-0266-51602-0001.fits>
- Archive: SDSS DR17 Science Archive Server (BOSS spectrum, plate 266, fibre 1)
- Terms: Public SDSS data; acknowledgement requested (sdss.org).
- SHA-256 of the fetched bytes: `c612b8d3609cfa830c444da60575f7093034682b71dcf8be89006f9f94e7f921`
- Covers: binary tables written by IDL's mwrfits: a 126-column one-row table of every scalar type and fixed strings.
- Cut:
  - HDU 0 (original HDU 0): as stored.
  - HDU 1 (original HDU 1): as stored.
  - HDU 2 (original HDU 2): as stored.
  - HDU 3 (original HDU 3): as stored.

## gaia-dr1-source.fits

- Source: <https://cdn.gea.esac.esa.int/Gaia/gdr1/gaia_source/fits/GaiaSource_000-000-000.fits>
- Archive: ESA Gaia Archive (Gaia DR1 gaia_source, first file)
- Terms: ESA/Gaia/DPAC, CC BY-SA 3.0 IGO; credit ESA/Gaia/DPAC.
- SHA-256 of the fetched bytes: `a4209e82c1468408ef7e5e74fc533d5e3fb5ee310e8df8d6567f2cc0341c4618`
- Covers: STIL's fits-plus layout: a primary byte array holding a VOTable, then a 57-column table with NaN-filled floats.
- Cut:
  - HDU 0 (original HDU 0): as stored.
  - HDU 1 (original HDU 1): rows 0-47 of 218453 (0-based); rewritten: NAXIS2.

## ps1-stack-rice.fits

- Source: <https://ps1images.stsci.edu/rings.v3.skycell/1784/059/rings.v3.skycell.1784.059.stk.g.unconv.fits>
- Archive: STScI Pan-STARRS1 image archive (DR2 stack, skycell 1784.059, g)
- Terms: Public Pan-STARRS1 data; acknowledgement requested (panstarrs.stsci.edu).
- SHA-256 of the fetched bytes: `b13559677a858d816db1388dcb3ac8a7ad16c12da8e3c96fbc0a961e27dd4dfb`
- Covers: cfitsio Rice tiles of int16 scaled by BSCALE and BZERO, one row per tile, with HIERARCH records.
- Cut:
  - HDU 0 (original HDU 0): as stored.
  - HDU 1 (original HDU 1): rows 3100-3107 of 6261 (0-based); heap bytes 32905074-32990989 kept, descriptors moved down by 32905074; rewritten: CRPIX2 NAXIS2 PCOUNT ZNAXIS2.

## ps1-mask-gzip.fits

- Source: <https://ps1images.stsci.edu/rings.v3.skycell/1784/059/rings.v3.skycell.1784.059.stk.g.unconv.mask.fits>
- Archive: STScI Pan-STARRS1 image archive (DR2 stack mask, skycell 1784.059, g)
- Terms: Public Pan-STARRS1 data; acknowledgement requested (panstarrs.stsci.edu).
- SHA-256 of the fetched bytes: `6e6d7f967b2e84749eaa2a0f02af354322c44f5de2e33bb987c1f8afd14b4a28`
- Covers: cfitsio GZIP_1 tiles of uint16, one row per tile.
- Cut:
  - HDU 0 (original HDU 0): as stored.
  - HDU 1 (original HDU 1): rows 3100-3107 of 6261 (0-based); heap bytes 339282-340135 kept, descriptors moved down by 339282; rewritten: CRPIX2 NAXIS2 PCOUNT ZNAXIS2.

## legacy-dr10-image.fits

- Source: <https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10/south/coadd/000/0001m002/legacysurvey-0001m002-image-g.fits.fz>
- Archive: NERSC Legacy Surveys DR10 (south coadd, brick 0001m002, g)
- Terms: Public Legacy Surveys data; acknowledgement requested (legacysurvey.org).
- SHA-256 of the fetched bytes: `c2069e7fc09524300e0a75e5901844a015435538509e484411deccc599499f19`
- Covers: fpack's RICE_ONE float tiles of 100x100, quantized with SUBTRACTIVE_DITHER_2, with ZSCALE and ZZERO columns.
- Cut:
  - HDU 0 (original HDU 0): as stored.
  - HDU 1 (original HDU 1): rows 665-667 of 1296 (0-based); heap bytes 5678557-5704102 kept, descriptors moved down by 5678557; rewritten: CRPIX1 CRPIX2 NAXIS2 PCOUNT ZDITHER0 ZNAXIS1 ZNAXIS2.

## legacy-dr10-maskbits.fits

- Source: <https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10/south/coadd/000/0001m002/legacysurvey-0001m002-maskbits.fits.fz>
- Archive: NERSC Legacy Surveys DR10 (south coadd, brick 0001m002, maskbits)
- Terms: Public Legacy Surveys data; acknowledgement requested (legacysurvey.org).
- SHA-256 of the fetched bytes: `e6a82a9cce746df80e970ca3938f1af89d3382195d402018696f7b044608a464`
- Covers: fpack's HCOMPRESS_1 tiles of int32 and uint8, a codec ymir does not decode.
- Cut:
  - HDU 0 (original HDU 0): as stored.
  - HDU 1 (original HDU 1): rows 55-56 of 1296 (0-based); heap bytes 11698-13230 kept, descriptors moved down by 11698; rewritten: CRPIX1 CRPIX2 NAXIS2 PCOUNT ZNAXIS1 ZNAXIS2.
  - HDU 2 (original HDU 2): rows 55-56 of 1296 (0-based); heap bytes 7387-7438 kept, descriptors moved down by 7387; rewritten: NAXIS2 PCOUNT ZNAXIS1 ZNAXIS2.

## nicer-rmf.fits

- Source: <https://heasarc.gsfc.nasa.gov/FTP/caldb/data/nicer/xti/cpf/rmf/nixtiref20170601v002.rmf>
- Archive: HEASARC CALDB (NICER XTI response matrix)
- Terms: NASA HEASARC CALDB, public.
- SHA-256 of the fetched bytes: `c7ba89ae22c7953ddffe8292593c466a3ddfc13945d6361abdf695886f65938e`
- Covers: an OGIP response matrix whose rows are heap arrays (1PI, 1PE), written by IDL.
- Cut:
  - HDU 0 (original HDU 0): as stored.
  - HDU 1 (original HDU 1): as stored.
  - HDU 2 (original HDU 2): rows 400-423 of 3451 (0-based); heap bytes 103980-111083 kept, descriptors moved down by 103980; rewritten: NAXIS2 PCOUNT DATASUM CHECKSUM.

## vla-uvfits.fits

- Source: <https://raw.githubusercontent.com/RadioAstronomySoftwareGroup/pyuvdata/v2.4.0/pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.uvfits>
- Archive: pyuvdata v2.4.0 test data (VLA TDEM0003, exported by CASA)
- Terms: pyuvdata, BSD 2-Clause; VLA data courtesy NRAO.
- SHA-256 of the fetched bytes: `78326b256344c17e502284425981a946d18466bfb3d22316226f0f69498fd701`
- Covers: random groups in the primary, then AIPS FQ, AN and WX tables.
- Cut:
  - HDU 0 (original HDU 0): groups 0-11 of 1360; rewritten: GCOUNT.
  - HDU 1 (original HDU 1): as stored.
  - HDU 2 (original HDU 2): as stored.
  - HDU 3 (original HDU 3): as stored.
