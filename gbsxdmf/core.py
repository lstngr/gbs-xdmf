#!/bin/env python
""" Generates an XDMF sidecar to GBS data.

This scripts allows the generation of XDMF-formatted metadata for GBS result
files. This enables loading GBS result data into programs supporting the XDMF
format, in particular Paraview.

This code is aimed at being very light and.thus does not rely on the GBS
library, gbspy. No extensive checks are performed.
"""
import os
import re
import argparse
import logging
import h5py
import lxml.etree as ET
import numpy as np
from natsort import natsorted

GBS_RESFILE_RE = r'^results_\d+.h5$'
GBS_CSTEP_RE = r'^\d+$'

NSMAP = {'xi':"http://www.w3.org/2001/XInclude"}
XDMFINFO = {"Version":"2.0"}


def get_gbs_files(path):
    '''Finds GBS results files at given path.

    Finds all HDF5 files that match the naming pattern of a GBS output, and
    returns a sorted list containing these. This routine does not attempt to
    open the said files, nor to check the consistency between them.

    PARAMETERS
    ==========
    path : str
        Path to the directory containing the result files.

    RETURNS
    =======
    list of str
        A sorted list of the result files that were found.
    '''
    regex = re.compile(GBS_RESFILE_RE)
    filenames = [f for f in os.listdir('.') if os.path.isfile(f)]
    matching = list(filter(r.match,filenames))
    matching = natsorted(matching)
    return matching

def get_legacy_meshgrid(file_name, r_0=None, r_scale=None):
    '''Builds meshgrid from coord[1-3] arrays.'''
    with h5py.File(file_name, 'r') as file_handle:
        x_c = file_handle['/data/var3d/theta/coord1'][:]
        y_c = file_handle['/data/var3d/theta/coord2'][:]
        z_c = file_handle['/data/var3d/theta/coord3'][:]

    logging.debug('{} {} {}'.format(file_name, r_0, r_scale))
    if r_0 and r_scale:
        logging.debug('Rescaling original GBS coordinates.')
        x_c *= r_0 / r_scale
        y_c *= r_0 / r_scale
        z_c *= r_0
    X, Y, Z = np.meshgrid(x_c, y_c, z_c)

    flip = lambda a: np.transpose(a, (2, 1, 0))
    X = flip(X)
    Y = flip(Y)
    Z = flip(Z)
    return X, Y, Z

def transform_torus(X, Y, Z, r_0):
    '''Make toroidal'''
    # Need to renormalize Z to close the torus
    # Z = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))
    # Z *= 2*np.pi
    X = r_0 + X - (np.max(X) - np.min(X)) / 2.
    if np.any(X<0.):
        logging.error(('Torus has non-physical dimensions from GBS ',
                       'normalization data. Data will be rescaled outwards.'))
        X = X - np.min(X) + r_0 / 1e3
    xcm = X*np.cos(Z)
    ycm = X*np.sin(Z)
    zcm = Y
    return xcm, ycm, zcm

def make_grid_sidecar(which_type, filename, prefix=''):
    '''Make a grid sidecar file'''
    major_radius = 1.0

    with h5py.File(filename, 'r') as file_handle:
        gbs_attrs_group = list(file_handle['data'].keys())[0]
        rorhos = file_handle['data/' + gbs_attrs_group].attrs['rorho_s']
    gbs_coords = get_legacy_meshgrid(filename, r_0=major_radius, r_scale=rorhos)

    if which_type == 'rectangle':
        # Fetch rectangle grid
        logging.debug('Fetching rectangular grid.')
    elif which_type == 'torus':
        gbs_coords = transform_torus(*gbs_coords, major_radius)
        logging.debug('Fetching toroidal grid.')
    else:
        raise ValueError('Grid type is not an allowed topology.')

    gbs_coords = np.stack(gbs_coords, axis=3)
    gbs_coords_shape = np.shape(gbs_coords)
    grid_info = {
            'nx': gbs_coords_shape[1],
            'ny': gbs_coords_shape[2],
            'nz': gbs_coords_shape[0],
            }

    if not prefix:
        prefix = ''
    if not prefix.endswith('_'):
        prefix += '_'
    sidecar_name = prefix + 'grid.h5'
    with h5py.File(sidecar_name, 'w') as file_handle:
        file_handle.create_dataset('XYZ', data=gbs_coords)
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

def write_gbs_function_xml(rootgrid, name, expr, fields, step_info, dimstr):
    '''Write an attribute field for a single GBS array'''
    attribute = ET.SubElement(rootgrid, 'Attribute',
            attrib={
                'Name':name,
                'AttributeType':'Scalar',
                'Center':'Node'
                })
    attrib_func = ET.SubElement(attribute, 'DataItem',
            attrib={
                'ItemType':'Function',
                'Function':expr,
                'Dimensions':dimstr
                })
    for field in fields:
        attrib_data = ET.SubElement(attrib_func, 'DataItem',
                attrib={
                    'Format':'HDF',
                    'NumberType':'Float',
                    'Precision':'8',
                    'Dimensions':dimstr
                    })
        attrib_data.text = \
            step_info['file']+':/data/var3d/'+field+'/'+step_info['id']

def make_xdmf(results, grid_sidecar, grid_info):
    '''Generates Xdmf sidecar'''
    gbs_info = extract_gbs_data(results)
    dimstr = '{} {} {}'.format(grid_info['nz'], grid_info['nx'],
            grid_info['ny'])
    grdstr = dimstr+' 3'

    sidecar_name = 'paraview_results_sidecar.xmf'
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

        write_gbs_field_xml(grid, 'omega', step_info, dimstr)
        write_gbs_field_xml(grid, 'psi', step_info, dimstr)
        write_gbs_field_xml(grid, 'strmf', step_info, dimstr)
        write_gbs_field_xml(grid, 'temperature', step_info, dimstr)
        write_gbs_field_xml(grid, 'temperaturi', step_info, dimstr)
        write_gbs_field_xml(grid, 'theta', step_info, dimstr)
        write_gbs_field_xml(grid, 'vpare', step_info, dimstr)
        write_gbs_field_xml(grid, 'vparepsi', step_info, dimstr)
        write_gbs_field_xml(grid, 'vpari', step_info, dimstr)

        write_gbs_function_xml(grid, 'n', 'exp($0)', ['theta'], step_info, dimstr)
        write_gbs_function_xml(grid, 'Te', 'exp($0)', ['temperature'], step_info, dimstr)
        write_gbs_function_xml(grid, 'Ti', 'exp($0)', ['temperaturi'], step_info, dimstr)
        write_gbs_function_xml(grid, 'pe', 'exp($0)*exp($1)', ['theta','temperature'], step_info, dimstr)
        write_gbs_function_xml(grid, 'pi', 'exp($0)*exp($1)', ['theta','temperaturi'], step_info, dimstr)

    #dtd = ET.DTD('Xdmf.dtd')
    #if not dtd.validate(root):
    #    print(dtd.error_log.filter_from_errors()[0])
    #    exit()

    tree = ET.ElementTree(root)
    #tree.write(sidecar_name, doctype='<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>')
    tree.write(sidecar_name)
    return sidecar_name


def main():
    """ Processes CLI options and generates XDMF data.

    This function is called whenever the file is run as a script. The steps this
    function take can be summarized as follow.
    1. Data from the simulation is gathered (available files, steps, basic
    domain size, etc.)
    2. A spatial grid is generated and saved to a HDF5 file.
    3. The GBS results are parsed (FIXME) for data and corresponding XDMF
    entries are established.
    """
    logging.basicConfig(level=0)
    parser = argparse.ArgumentParser(
        description='Generate XDMF sidecar files for GBS simulations.')
    parser.add_argument(
        '--topology', metavar='type', type=str, default='rectangle',
        choices=('rectangle', 'torus'), help=('topology of the generated grid '
                                              '(rectangle, torus)'))
    parser.add_argument('--prefix', type=str, default='xdmf',
                        help='prefix to the output files\' names.')
    parser.add_argument('path', type=str, default=os.getcwd(),
                        help='Path containing the GBS results.')
    args = parser.parse_args()

    gbs_results = get_gbs_files(args.path)
    grid_fname, grid_info = make_grid_sidecar(args.topology, gbs_results[0],
                                              prefix=args.prefix)
    make_xdmf(gbs_results, grid_fname, grid_info)


if __name__=='__main__':
    main()
