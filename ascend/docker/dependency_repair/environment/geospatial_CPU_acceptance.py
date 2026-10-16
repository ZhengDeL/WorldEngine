"""Actual CRS APIs for three libraries, including four nuPlan city UTM systems."""
import hashlib
import math
import os
from pathlib import Path

DATA=Path('/opt/worldengine-geo/share/proj')
DATABASE_SHA='42f1f5efade59a807bfb58cf444b0d63a341f4633aca1401e64406d3d421c562'
CITIES=[('Boston',32619,-71.0589,42.3601),('LasVegas',32611,-115.1398,36.1699),
 ('Singapore',32648,103.8198,1.3521),('Pittsburgh',32617,-79.9959,40.4406)]

def run_checks():
 if os.environ.get('PROJ_DATA')!=str(DATA) or os.environ.get('PROJ_LIB')!=str(DATA):raise RuntimeError('Actual declared GDAL/PROJ data directory not selected')
 database=DATA/'proj.db'
 if database.stat().st_size!=9261056 or hashlib.sha256(database.read_bytes()).hexdigest()!=DATABASE_SHA:raise RuntimeError('Original PROJ9.5.1 data bytes changed')
 # Preserve the original failing same-process import order in this regression gate.
 import pyproj
 import rasterio
 import fiona
 from rasterio.crs import CRS as RasterCRS
 from rasterio.warp import transform as raster_transform
 from fiona.crs import CRS as FionaCRS
 from fiona.transform import transform as fiona_transform
 def py(source,target,lon,lat):return pyproj.Transformer.from_crs(source,target,always_xy=True).transform(lon,lat)
 def ras(source,target,lon,lat):
  x,y=raster_transform(source,target,[lon],[lat]);return x[0],y[0]
 def fio(source,target,lon,lat):
  x,y=fiona_transform(source,target,[lon],[lat]);return x[0],y[0]
 checks={};references={}
 for name,crs,transform in [('pyproj',pyproj.CRS,py),('rasterio',RasterCRS,ras),('fiona',FionaCRS,fio)]:
  rows={}
  for city,epsg,lon,lat in CITIES:
   if crs.from_epsg(epsg).to_epsg()!=epsg:raise RuntimeError(name+' EPSG lookup failed: '+city)
   x,y=transform('EPSG:4326','EPSG:'+str(epsg),lon,lat)
   a,b=transform('EPSG:'+str(epsg),'EPSG:4326',x,y)
   error=max(abs(a-lon),abs(b-lat))
   if not all(math.isfinite(z)for z in (x,y,a,b)) or not 100000<x<900000 or not 0<y<10000000 or error>1e-8:
    raise RuntimeError(name+' real UTM roundtrip failed: '+city)
   if name=='pyproj':references[city]=(x,y)
   cross=max(abs(x-references[city][0]),abs(y-references[city][1]))
   if cross>1e-4:raise RuntimeError(name+' independent library UTM disagreement: '+city)
   rows[city]={'EPSG':epsg,'longitude_latitude':[lon,lat],'UTM':[x,y],
    'roundtrip_max_degrees':error,'library_difference_from_pyproj_metres':cross}
  lon,lat=116.3,39.9
  x,y=transform('EPSG:4326','EPSG:3857',lon,lat)
  expected_x=6378137*math.radians(lon);expected_y=6378137*math.log(math.tan(math.pi/4+math.radians(lat)/2))
  mathematical=max(abs(x-expected_x),abs(y-expected_y))
  a,b=transform('EPSG:3857','EPSG:4326',x,y)
  if mathematical>1e-5 or max(abs(a-lon),abs(b-lat))>1e-8:raise RuntimeError(name+' independent WebMercator formula/roundtrip failed')
  rows['WebMercator']={'EPSG':3857,'forward':[x,y],'formula_max_error_metres':mathematical,
   'roundtrip_max_degrees':max(abs(a-lon),abs(b-lat))}
  checks[name]=rows
 return {'result':'pass_actual_three_library_CRS_UTM_and_independent_WebMercator_APIs',
  'PROJ_DATA':str(DATA),'database_sha256':DATABASE_SHA,'versions':{
   'pyproj':pyproj.__version__,'pyproj_PROJ':pyproj.proj_version_str,
   'rasterio':rasterio.__version__,'rasterio_GDAL':rasterio.__gdal_version__,
   'rasterio_PROJ':rasterio.__proj_version__,'fiona':fiona.__version__,'fiona_GDAL':fiona.__gdal_version__},
  'pyproj_actual_data_dir':pyproj.datadir.get_data_dir(),'checks':checks}
