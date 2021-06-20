import h5py, os, re, xml
import lxml.etree as ET
import numpy as np
from natsort import natsorted

GBS_RESFILE_RE = '^results_\d+.h5$'
GBS_CSTEP_RE = '^\d+$'

NSMAP = {'xi':"http://www.w3.org/2001/XInclude"}
XDMFINFO = {"Version":"3.0"}

def get_gbs_files(path='.'):
    '''Finds GBS results files at given path.'''
    rfiles = []
    r = re.compile(GBS_RESFILE_RE)
    for dirpath, dirnames, filenames in os.walk(path):
        matching = list(filter(r.match,filenames))
    return matching

def get_legacy_meshgrid(fname):
    '''Builds meshgrid from coord[1-3] arrays.'''
    with h5py.File(fname, 'r') as f:
        xc = f['/data/var3d/theta/coord1'][:]
        yc = f['/data/var3d/theta/coord2'][:]
        zc = f['/data/var3d/theta/coord3'][:]
    X, Y, Z = np.meshgrid(xc, yc, zc)
    flip = lambda a: np.transpose(a, (2, 1, 0))
    X = flip(X)
    Y = flip(Y)
    Z = flip(Z)
    return X, Y, Z

def grid_post(X, Y, Z):
    '''Make toroidal'''
    R0 = 400
    Z = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))
    Z *= 2*np.pi
    XCM = (X+R0)*np.cos(Z)
    YCM = (X+R0)*np.sin(Z)
    ZCM = Y
    X, Y, Z = XCM, YCM, ZCM
    return X, Y, Z

def get_gbs_meshgrid(fname=None):
    '''Extracts a meshgrid from given result file.'''
    X, Y, Z = get_legacy_meshgrid(fname)
    X, Y, Z = grid_post(X, Y, Z) # Transform grid if needed
    M = np.stack((X, Y, Z), axis=3)
    return M

def make_grid_sidecar(fname=None):
    '''Make a grid sidecar file'''
    sidecar_name = 'paraview_grid_sidecar.h5'
    M = get_gbs_meshgrid(fname)
    N = np.shape(M)
    grid_info = {
            'nx':N[1],
            'ny':N[2],
            'nz':N[0]
            }
    with h5py.File(sidecar_name, 'w') as f:
        f.create_dataset('XYZ', data=M)
    return sidecar_name, grid_info

def extract_gbs_data(results):
    '''Returns a list of dict describing each time step'''
    info = []
    for fname in results:
        with h5py.File(fname, 'r') as f:
            times = f['/data/var3d/time'][:]
            steps = f['/data/var3d/theta'].keys()
            r = re.compile(GBS_CSTEP_RE)
            steps = list(filter(r.match,steps))
        for t, s in zip(times, steps):
            step_info = {'id':s,'time':t,'file':fname}
            info.append(step_info.copy())
    return info

def write_gbs_field_xml(rootgrid, field, step_info, dimstr):
    '''Write an attribute field for a single GBS array'''
    attribute = ET.SubElement(rootgrid, 'Attribute',
            attrib={
                'Name':field,
                'AttributeType':'Scalar',
                'Center':'Node'
                })
    attrib_data = ET.SubElement(attribute, 'DataItem',
            attrib={
                'Format':'HDF',
                'NumberType':'Float',
                'Precision':'8',
                'Dimensions':dimstr
                })
    attrib_data.text = \
        step_info['file']+':/data/var3d/'+field+'/'+step_info['id']
    return None

def make_xdmf(results, grid_sidecar, grid_info):
    '''Generates Xdmf sidecar'''
    gbs_info = extract_gbs_data(results)
    dimstr = '{} {} {}'.format(grid_info['nz'], grid_info['nx'],
            grid_info['ny'])
    grdstr = dimstr+' 3'

    sidecar_name = 'paraview_results_sidecar.xmf3'
    root = ET.Element("Xdmf", attrib=XDMFINFO, nsmap=NSMAP)
    domain = ET.SubElement(root, 'Domain')
    temporal = ET.SubElement(domain, 'Grid',
            attrib={
                'Name':'GBS_collection',
                'GridType':'Collection',
                'CollectionType':'Temporal'
                })

    for step_info in gbs_info:
        grid = ET.SubElement(temporal, 'Grid',
                attrib={
                    'Name':'grid'+str(step_info['id']),
                    'GridType':'Uniform'
                    })
        time = ET.SubElement(grid, 'Time',
                attrib={'Value':str(step_info['time'])})
        topology = ET.SubElement(grid, 'Topology',
                attrib={
                    'TopologyType':'3DSMesh',
                    'Dimensions':dimstr})
        geometry = ET.SubElement(grid, 'Geometry',
                attrib={
                    'GeometryType':'XYZ'
                    })
        geo_data = ET.SubElement(geometry, 'DataItem',
                attrib={
                    'Format':'HDF',
                    'NumberType':'Float',
                    'Precision':'8',
                    'Dimensions':grdstr
                    })
        geo_data.text = grid_sidecar+':/XYZ'

        write_gbs_field_xml(grid, 'theta', step_info, dimstr)
        write_gbs_field_xml(grid, 'strmf', step_info, dimstr)
        write_gbs_field_xml(grid, 'omega', step_info, dimstr)
        write_gbs_field_xml(grid, 'temperature', step_info, dimstr)
        write_gbs_field_xml(grid, 'temperaturi', step_info, dimstr)
        write_gbs_field_xml(grid, 'vpare', step_info, dimstr)
        write_gbs_field_xml(grid, 'vpari', step_info, dimstr)

    dtd = ET.DTD('Xdmf.dtd')
    if not dtd.validate(root):
        print(dtd.error_log.filter_from_errors()[0])
        exit()

    tree = ET.ElementTree(root)
    tree.write(sidecar_name, doctype='<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>')
    return sidecar_name

if __name__=='__main__':
    gbs_results = get_gbs_files()
    grid_fname, grid_info = make_grid_sidecar(gbs_results[0])
    make_xdmf(gbs_results, grid_fname, grid_info)

