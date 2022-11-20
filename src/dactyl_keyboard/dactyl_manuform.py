import argparse
from collections.abc import Iterable
import copy
import json
import logging
import math
import os
import os.path as path
import pathlib
import sys
from typing import Any, Optional

if sys.version_info[:2] > (3, 9):
    import importlib.resources as resources
else:
    import importlib_resources as resources

import numpy as np

from . engines.engine import GeometryEngine
from . generate_configuration import GenerateConfigAction, shape_config


class LogLevelAction(argparse.Action):
    """
    Set the log level
    """

    log_levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }

    def __init__(self, option_strings, dest: str, **kwargs):
        kwargs.update({
            'default': "INFO",
            'type': str,
            'choices': self.log_levels.keys(),
            'help': "The log level to use."
        })
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, _parser: argparse.ArgumentParser, namespace: argparse.Namespace, values: pathlib.Path, _option_string: Optional[str] = None):
        setattr(namespace, self.dest, values)


parser = argparse.ArgumentParser(description="Generate a dactyl keyboard.")
parser.add_argument("--generate-config", action=GenerateConfigAction)
parser.add_argument("--config", default=argparse.SUPPRESS, type=pathlib.Path, help="A config file to control keyboard generation.")
parser.add_argument("--log-level", action=LogLevelAction)
args = parser.parse_args()

logging.basicConfig(level=args.log_level)

###############################################
# EXTREMELY UGLY BUT FUNCTIONAL BOOTSTRAP
###############################################

# IMPORT DEFAULT CONFIG IN CASE NEW PARAMETERS EXIST
for key, item in shape_config.items():
    locals()[key] = item

if not "config" in args:
    logging.info("NO CONFIGURATION SPECIFIED, USING DEFAULT CONFIGURATION")
else:
    with open(args.config, mode="rt", encoding="utf-8") as fid:
        for key, item in json.load(fid).items():
            locals()[key] = item

logging.info("Using engine %s", ENGINE)

if save_dir in ['', None, '.']:
    save_path = path.join(r".", "things")
else:
    save_path = path.join(r".", "things", save_dir)

parts_path = resources.files("dactyl_keyboard.parts")

###############################################
# END EXTREMELY UGLY BOOTSTRAP
###############################################

####################################################
# HELPER FUNCTIONS TO MERGE CADQUERY AND OPENSCAD
####################################################

geometry_engine: Optional[GeometryEngine] = None
if ENGINE == 'cadquery':
    from . engines.cadquery_engine import CadQueryEngine
    geometry_engine = CadQueryEngine()
else:
    from . helpers_solid import *

####################################################
# END HELPER FUNCTIONS
####################################################


debug_exports = False


if oled_mount_type is not None and oled_mount_type != "NONE":
    for item in oled_configurations[oled_mount_type]:
        locals()[item] = oled_configurations[oled_mount_type][item]

if nrows > 5:
    column_style = column_style_gt5

centerrow = nrows - centerrow_offset

lastrow = nrows - 1
if reduced_outer_cols > 0 or reduced_inner_cols > 0:
    cornerrow = lastrow - 1
else:
    cornerrow = lastrow
lastcol = ncols - 1


# Derived values
if plate_style in ['NUB', 'HS_NUB']:
    keyswitch_height = nub_keyswitch_height
    keyswitch_width = nub_keyswitch_width
elif plate_style in ['UNDERCUT', 'HS_UNDERCUT', 'NOTCH', 'HS_NOTCH']:
    keyswitch_height = undercut_keyswitch_height
    keyswitch_width = undercut_keyswitch_width
else:
    keyswitch_height = hole_keyswitch_height
    keyswitch_width = hole_keyswitch_width

if 'HS_' in plate_style:
    symmetry = "asymmetric"
    plate_file = "hot_swap_plate"
    plate_offset = 0.0

if (trackball_in_wall or ('TRACKBALL' in thumb_style)) and not ball_side == 'both':
    symmetry = "asymmetric"

mount_width = keyswitch_width + 2 * plate_rim
mount_height = keyswitch_height + 2 * plate_rim
mount_thickness = plate_thickness

if default_1U_cluster and thumb_style == 'DEFAULT':
    double_plate_height = (.7*sa_double_length - mount_height) / 3
elif thumb_style == 'DEFAULT':
    double_plate_height = (.95*sa_double_length - mount_height) / 3
else:
    double_plate_height = (sa_double_length - mount_height) / 3


if oled_mount_type is not None and oled_mount_type != "NONE":
    left_wall_x_offset = oled_left_wall_x_offset_override
    left_wall_z_offset = oled_left_wall_z_offset_override
    left_wall_lower_y_offset = oled_left_wall_lower_y_offset
    left_wall_lower_z_offset = oled_left_wall_lower_z_offset


cap_top_height = plate_thickness + sa_profile_key_height
row_radius = ((mount_height + extra_height) / 2) / (math.sin(alpha / 2)) + cap_top_height
column_radius = (
    ((mount_width + extra_width) / 2) / (math.sin(beta / 2))
) + cap_top_height
column_x_delta = -1 - column_radius * math.sin(beta)
column_base_angle = beta * (centercol - 2)


teensy_width = 20
teensy_height = 12
teensy_length = 33
teensy2_length = 53
teensy_pcb_thickness = 2
teensy_offset_height = 5
teensy_holder_top_length = 18
teensy_holder_width = 7 + teensy_pcb_thickness
teensy_holder_height = 6 + teensy_width


# save_path = path.join("..", "things", save_dir)
if not path.isdir(save_path):
    os.mkdir(save_path)


def triangle_hulls(shapes: Iterable) -> Any:
    hulls = []
    for i in range(len(shapes) - 2):
        hulls.append(geometry_engine.convex_hull(shapes[i: (i + 3)]))

    return geometry_engine.union(hulls)


def column_offset(column: int) -> list:
    return column_offsets[column]

# column_style='fixed'


def single_plate(cylinder_segments=100, side="right"):

    if plate_style in ['NUB', 'HS_NUB']:
        tb_border = (mount_height-keyswitch_height)/2
        top_wall = geometry_engine.box(mount_width, tb_border, plate_thickness)
        top_wall = geometry_engine.translate(top_wall, (0, (tb_border / 2) + (keyswitch_height / 2), plate_thickness / 2))

        lr_border = (mount_width - keyswitch_width) / 2
        left_wall = geometry_engine.box(lr_border, mount_height, plate_thickness)
        left_wall = geometry_engine.translate(left_wall, ((lr_border / 2) + (keyswitch_width / 2), 0, plate_thickness / 2))

        side_nub = geometry_engine.cylinder(radius=1, height=2.75)
        side_nub = geometry_engine.rotate(side_nub, (90, 0, 0))
        side_nub = geometry_engine.translate(side_nub, (keyswitch_width / 2, 0, 1))

        nub_cube = geometry_engine.box(1.5, 2.75, plate_thickness)
        nub_cube = geometry_engine.translate(nub_cube, ((1.5 / 2) + (keyswitch_width / 2),  0, plate_thickness / 2))

        side_nub2 = geometry_engine.convex_hull((side_nub, nub_cube))
        side_nub2 = geometry_engine.union([side_nub2, side_nub, nub_cube])

        plate_half1 = geometry_engine.union([top_wall, left_wall, side_nub2])
        plate_half2 = plate_half1
        plate_half2 = geometry_engine.mirror(plate_half2, 'XZ')
        plate_half2 = geometry_engine.mirror(plate_half2, 'YZ')

        plate = geometry_engine.union([plate_half1, plate_half2])

    else:  # 'HOLE' or default, square cutout for non-nub designs.
        plate = geometry_engine.box(mount_width, mount_height, mount_thickness)
        plate = geometry_engine.translate(plate, (0.0, 0.0, mount_thickness / 2.0))

        shape_cut = geometry_engine.box(keyswitch_width, keyswitch_height, mount_thickness * 2 + .02)
        shape_cut = geometry_engine.translate(shape_cut, (0.0, 0.0, mount_thickness-.01))

        plate = geometry_engine.difference(plate, [shape_cut])

    if plate_style in ['UNDERCUT', 'HS_UNDERCUT', 'NOTCH', 'HS_NOTCH']:
        if plate_style in ['UNDERCUT', 'HS_UNDERCUT']:
            undercut = geometry_engine.box(
                keyswitch_width + 2 * clip_undercut,
                keyswitch_height + 2 * clip_undercut,
                mount_thickness
            )

        if plate_style in ['NOTCH', 'HS_NOTCH']:
            undercut = geometry_engine.box(
                notch_width,
                keyswitch_height + 2 * clip_undercut,
                mount_thickness
            )
            undercut = geometry_engine.union([undercut,
                                              geometry_engine.box(
                                                  keyswitch_width + 2 * clip_undercut,
                                                  notch_width,
                                                  mount_thickness
                                              )
                                              ])

        undercut = geometry_engine.translate(undercut, (0.0, 0.0, -clip_thickness + mount_thickness / 2.0))

        if ENGINE == 'cadquery' and undercut_transition > 0:
            undercut = undercut.faces("+Z").chamfer(undercut_transition, clip_undercut)

        plate = geometry_engine.difference(plate, [undercut])

    if plate_file is not None:
        plate_path = pathlib.Path(plate_file)
        plate_path_importer = next([importer for importer in geometry_engine.importers() if importer.file_type() == plate_path.suffix], None)
        if plate_path.exists() and plate_path_importer is not None:
            socket = plate_path_importer.import_resource(plate_path)
        else:
            for importer in geometry_engine.importers():
                importable_plate_file = parts_path.joinpath(plate_file + importer.file_type())
                if importable_plate_file.is_file():
                    with resources.as_file(importable_plate_file) as resource_plate:
                        socket = importer.import_resource(resource_plate)
                        break

        socket = geometry_engine.translate(socket, [0, 0, plate_thickness + plate_offset])
        plate = geometry_engine.union([plate, socket])

    if plate_holes:
        half_width = plate_holes_width/2.
        half_height = plate_holes_height/2.
        x_off = plate_holes_xy_offset[0]
        y_off = plate_holes_xy_offset[1]
        holes = [
            geometry_engine.translate(
                geometry_engine.cylinder(radius=plate_holes_diameter/2, height=plate_holes_depth+.01),
                (x_off+half_width, y_off+half_height, plate_holes_depth/2-.01)
            ),
            geometry_engine.translate(
                geometry_engine.cylinder(radius=plate_holes_diameter / 2, height=plate_holes_depth+.01),
                (x_off-half_width, y_off+half_height, plate_holes_depth/2-.01)
            ),
            geometry_engine.translate(
                geometry_engine.cylinder(radius=plate_holes_diameter / 2, height=plate_holes_depth+.01),
                (x_off-half_width, y_off-half_height, plate_holes_depth/2-.01)
            ),
            geometry_engine.translate(
                geometry_engine.cylinder(radius=plate_holes_diameter / 2, height=plate_holes_depth+.01),
                (x_off+half_width, y_off-half_height, plate_holes_depth/2-.01)
            ),
        ]
        plate = geometry_engine.difference(plate, holes)

    if side == "left":
        plate = geometry_engine.mirror(plate, np.array((1, 0, 0)))

    return plate


def plate_pcb_cutout(side="right"):
    shape = geometry_engine.box(*plate_pcb_size)
    shape = geometry_engine.translate(shape, (0, 0, -plate_pcb_size[2]/2))
    shape = geometry_engine.translate(shape, plate_pcb_offset)

    if side == "left":
        shape = geometry_engine.mirror(shape, np.array((1, 0, 0)))

    return shape


def trackball_cutout(segments=100, side="right"):
    if trackball_modular:
        hole_diameter = ball_diameter + 2 * (ball_gap + ball_wall_thickness + trackball_modular_clearance+trackball_modular_lip_width)-.1
        shape = geometry_engine.cylinder(hole_diameter / 2, trackball_hole_height)
    else:
        shape = geometry_engine.cylinder(trackball_hole_diameter / 2, trackball_hole_height)
    return shape


def trackball_socket(segments=100, side="right"):
    if trackball_modular:
        hole_diameter = ball_diameter + 2 * (ball_gap + ball_wall_thickness + trackball_modular_clearance)
        ring_diameter = hole_diameter + 2 * trackball_modular_lip_width
        ring_height = trackball_modular_ring_height
        ring_z_offset = mount_thickness - trackball_modular_ball_height
        shape = geometry_engine.cylinder(ring_diameter / 2, ring_height)
        shape = geometry_engine.translate(shape, (0, 0, -ring_height / 2 + ring_z_offset))

        cutter = geometry_engine.cylinder(hole_diameter / 2, ring_height + .2)
        cutter = geometry_engine.translate(cutter, (0, 0, -ring_height / 2 + ring_z_offset))

        sensor = None

    else:
        shape = import_file(parts_path, "trackball_socket_body_34mm")
        sensor = import_file(parts_path, "trackball_sensor_mount")
        cutter = import_file(parts_path, "trackball_socket_cutter_34mm")
        cutter = geometry_engine.union([cutter, import_file(parts_path, "trackball_sensor_cutter")])

    # return shape, cutter
    return shape, cutter, sensor


def trackball_ball(segments=100, side="right"):
    shape = geometry_engine.sphere(ball_diameter / 2)
    return shape

################
## SA Keycaps ##
################


def keycap(*args, **kwargs):
    if show_caps == 'CHOC':
        return choc_cap(*args, **kwargs)
    elif show_caps == 'MX':
        return sa_cap(*args, **kwargs)
    else:
        return sa_cap(*args, **kwargs)


def sa_cap(Usize=1):
    # MODIFIED TO NOT HAVE THE ROTATION.  NEEDS ROTATION DURING ASSEMBLY
    # sa_length = 18.25

    if Usize == 1:
        bl2 = 18.5/2
        bw2 = 18.5/2
        m = 17 / 2
        pl2 = 6
        pw2 = 6

    elif Usize == 2:
        bl2 = sa_length
        bw2 = sa_length / 2
        m = 0
        pl2 = 16
        pw2 = 6

    elif Usize == 1.5:
        bl2 = sa_length / 2
        bw2 = 27.94 / 2
        m = 0
        pl2 = 6
        pw2 = 11

    k1 = geometry_engine.box(bw2 * 2, bl2 * 2, 0.1)
    k1 = geometry_engine.translate(k1, (0, 0, 0.05))
    k2 = geometry_engine.box(pw2 * 2, pl2 * 2, 0.1)
    k2 = geometry_engine.translate(k2, (0, 0, 12.0))
    if m > 0:
        m1 = geometry_engine.box(m * 2, m * 2, 0.1)
        m1 = geometry_engine.translate(m1, (0, 0, 6.0))
        key_cap = geometry_engine.convex_hull([k1, k2, m1])
    else:
        key_cap = geometry_engine.convex_hull([k1, k2])

    key_cap = geometry_engine.translate(key_cap, (0, 0, 5 + plate_thickness))

    if show_pcbs:
        key_cap = add([key_cap, key_pcb()])

    return key_cap


def choc_cap(Usize=1):
    # MODIFIED TO NOT HAVE THE ROTATION.  NEEDS ROTATION DURING ASSEMBLY
    # sa_length = 18.25

    if Usize == 1:
        bl2 = 18.0/2
        bw2 = 17.5/2
        bt = 2
        pl2 = 15.0/2
        pw2 = 14.5/2
        pt = 1.5
        gap = 1.5

    k_box = geometry_engine.box(bw2 * 2, bl2 * 2, 0.1)
    k1 = geometry_engine.translate(k_box, (0, 0, 0.05))
    k2 = geometry_engine.translate(k_box, (0, 0, 0.05+bt))
    k3 = geometry_engine.box(pw2 * 2, pl2 * 2, 0.1)
    k3 = geometry_engine.translate(k3, (0, 0, 0.05+bt+pt))
    key_cap = geometry_engine.convex_hull((k1, k2, k3))

    key_cap = geometry_engine.translate(key_cap, (0, 0, 2.8 + plate_thickness))

    if show_pcbs:
        key_cap = add([key_cap, key_pcb()])

    return key_cap


def key_pcb():
    shape = geometry_engine.box(pcb_width, pcb_height, pcb_thickness)
    shape = geometry_engine.translate(shape, (0, 0, -pcb_thickness/2))
    hole = geometry_engine.cylinder(pcb_hole_diameter/2, pcb_thickness+.2)
    hole = geometry_engine.translate(hole, (0, 0, -(pcb_thickness+.1)/2))
    holes = [
        geometry_engine.translate(hole, (pcb_hole_pattern_width/2, pcb_hole_pattern_height/2, 0)),
        geometry_engine.translate(hole, (-pcb_hole_pattern_width / 2, pcb_hole_pattern_height / 2, 0)),
        geometry_engine.translate(hole, (-pcb_hole_pattern_width / 2, -pcb_hole_pattern_height / 2, 0)),
        geometry_engine.translate(hole, (pcb_hole_pattern_width / 2, -pcb_hole_pattern_height / 2, 0)),
    ]
    shape = geometry_engine.difference(shape, holes)

    return shape

#########################
## Placement Functions ##
#########################


def rotate_around_x(position, angle):
    t_matrix = np.array(
        [
            [1, 0, 0],
            [0, math.cos(angle), -math.sin(angle)],
            [0, math.sin(angle), math.cos(angle)],
        ]
    )
    return np.matmul(t_matrix, position)


def rotate_around_y(position, angle):
    t_matrix = np.array(
        [
            [math.cos(angle), 0, math.sin(angle)],
            [0, 1, 0],
            [-math.sin(angle), 0, math.cos(angle)],
        ]
    )
    return np.matmul(t_matrix, position)


def apply_key_geometry(
        shape,
        translate_fn,
        rotate_x_fn,
        rotate_y_fn,
        column,
        row,
        column_style=column_style,
):

    logging.debug("apply_key_geometry()")

    column_angle = beta * (centercol - column)

    if column_style == "orthographic":
        column_z_delta = column_radius * (1 - math.cos(column_angle))
        shape = translate_fn(shape, [0, 0, -row_radius])
        shape = rotate_x_fn(shape, alpha * (centerrow - row))
        shape = translate_fn(shape, [0, 0, row_radius])
        shape = rotate_y_fn(shape, column_angle)
        shape = translate_fn(
            shape, [-(column - centercol) * column_x_delta, 0, column_z_delta]
        )
        shape = translate_fn(shape, column_offset(column))

    elif column_style == "fixed":
        shape = rotate_y_fn(shape, fixed_angles[column])
        shape = translate_fn(shape, [fixed_x[column], 0, fixed_z[column]])
        shape = translate_fn(shape, [0, 0, -(row_radius + fixed_z[column])])
        shape = rotate_x_fn(shape, alpha * (centerrow - row))
        shape = translate_fn(shape, [0, 0, row_radius + fixed_z[column]])
        shape = rotate_y_fn(shape, fixed_tenting)
        shape = translate_fn(shape, [0, column_offset(column)[1], 0])

    else:
        shape = translate_fn(shape, [0, 0, -row_radius])
        shape = rotate_x_fn(shape, alpha * (centerrow - row))
        shape = translate_fn(shape, [0, 0, row_radius])
        shape = translate_fn(shape, [0, 0, -column_radius])
        shape = rotate_y_fn(shape, column_angle)
        shape = translate_fn(shape, [0, 0, column_radius])
        shape = translate_fn(shape, column_offset(column))

    shape = rotate_y_fn(shape, tenting_angle)
    shape = translate_fn(shape, [0, 0, keyboard_z_offset])

    return shape


def x_rot(shape, angle):
    return geometry_engine.rotate(shape, [math.degrees(angle), 0, 0])


def y_rot(shape, angle):
    return geometry_engine.rotate(shape, [0, math.degrees(angle), 0])


def key_place(shape, column, row):
    logging.debug("key_place()")
    return apply_key_geometry(shape, geometry_engine.translate, x_rot, y_rot, column, row)


# def translate(shape, xyz):
#     logging.debug("translate()")
#     vals = []
#     for i in range(len(shape)):
#         vals.append(shape[i] + xyz[i])
#     return vals


def key_position(position, column, row):
    logging.debug("key_position()")
    return apply_key_geometry(
        position, geometry_engine.translate, rotate_around_x, rotate_around_y, column, row
    )


def key_holes(side="right"):
    logging.debug("key_holes()")
    # hole = single_plate()
    holes = []
    for column in range(ncols):
        for row in range(nrows):
            if (reduced_inner_cols <= column < (ncols - reduced_outer_cols)) or (not row == lastrow):
                holes.append(key_place(single_plate(side=side), column, row))

    shape = geometry_engine.union(holes)

    return shape


def plate_pcb_cutouts(side="right"):
    logging.debug("plate_pcb_cutouts()")
    # hole = single_plate()
    cutouts = []
    for column in range(ncols):
        for row in range(nrows):
            if (reduced_inner_cols <= column < (ncols - reduced_outer_cols)) or (not row == lastrow):
                cutouts.append(key_place(plate_pcb_cutout(side=side), column, row))

    # cutouts = geometry_engine.union(cutouts)

    return cutouts


def caps(cap_type="MX"):
    caps = None

    for column in range(ncols):
        for row in range(nrows):
            if (reduced_inner_cols <= column < (ncols - reduced_outer_cols)) or (not row == lastrow):
                if caps is None:
                    caps = key_place(keycap(), column, row)
                else:
                    caps = add([caps, key_place(keycap(), column, row)])

    return caps


####################
## Web Connectors ##
####################


def web_post():
    logging.debug("web_post()")
    post = geometry_engine.box(post_size, post_size, web_thickness)
    post = geometry_engine.translate(post, (0, 0, plate_thickness - (web_thickness / 2)))
    return post


def web_post_tr(wide=False):
    if wide:
        w_divide = 1.2
    else:
        w_divide = 2.0

    return geometry_engine.translate(web_post(), ((mount_width / w_divide) - post_adj, (mount_height / 2) - post_adj, 0))


def web_post_tl(wide=False):
    if wide:
        w_divide = 1.2
    else:
        w_divide = 2.0
    return geometry_engine.translate(web_post(), (-(mount_width / w_divide) + post_adj, (mount_height / 2) - post_adj, 0))


def web_post_bl(wide=False):
    if wide:
        w_divide = 1.2
    else:
        w_divide = 2.0
    return geometry_engine.translate(web_post(), (-(mount_width / w_divide) + post_adj, -(mount_height / 2) + post_adj, 0))


def web_post_br(wide=False):
    if wide:
        w_divide = 1.2
    else:
        w_divide = 2.0
    return geometry_engine.translate(web_post(), ((mount_width / w_divide) - post_adj, -(mount_height / 2) + post_adj, 0))


def connectors():
    logging.debug("connectors()")
    hulls = []
    for column in range(ncols - 1):
        if reduced_inner_cols <= column < (ncols - reduced_outer_cols-1):
            iterrows = lastrow+1
        else:
            iterrows = lastrow
        for row in range(iterrows):  # need to consider last_row?
            # for row in range(nrows):  # need to consider last_row?
            places = []
            places.append(key_place(web_post_tl(), column + 1, row))
            places.append(key_place(web_post_tr(), column, row))
            places.append(key_place(web_post_bl(), column + 1, row))
            places.append(key_place(web_post_br(), column, row))
            hulls.append(triangle_hulls(places))

    for column in range(ncols):
        if reduced_inner_cols <= column < (ncols - reduced_outer_cols):
            iterrows = lastrow
        else:
            iterrows = cornerrow
        for row in range(iterrows):
            places = []
            places.append(key_place(web_post_bl(), column, row))
            places.append(key_place(web_post_br(), column, row))
            places.append(key_place(web_post_tl(), column, row + 1))
            places.append(key_place(web_post_tr(), column, row + 1))
            hulls.append(triangle_hulls(places))

    for column in range(ncols - 1):
        if (reduced_inner_cols <= column < (ncols - reduced_outer_cols-1)):
            iterrows = lastrow
        else:
            iterrows = cornerrow
        for row in range(iterrows):
            places = []
            places.append(key_place(web_post_br(), column, row))
            places.append(key_place(web_post_tr(), column, row + 1))
            places.append(key_place(web_post_bl(), column + 1, row))
            places.append(key_place(web_post_tl(), column + 1, row + 1))
            hulls.append(triangle_hulls(places))

        if column == (reduced_inner_cols-1):
            places = []
            places.append(key_place(web_post_bl(), column + 1, iterrows))
            places.append(key_place(web_post_br(), column, iterrows))
            places.append(key_place(web_post_tl(), column + 1, iterrows + 1))
            places.append(key_place(web_post_bl(), column + 1, iterrows + 1))
            hulls.append(triangle_hulls(places))
        if column == (ncols - reduced_outer_cols - 1):
            places = []
            places.append(key_place(web_post_br(), column, iterrows))
            places.append(key_place(web_post_bl(), column + 1, iterrows))
            places.append(key_place(web_post_tr(), column, iterrows + 1))
            places.append(key_place(web_post_br(), column, iterrows + 1))
            hulls.append(triangle_hulls(places))

    return geometry_engine.union(hulls)
    # return add(hulls)


############
## Thumbs ##
############


def thumborigin():
    corner = cornerrow if reduced_inner_cols > 0 else lastrow
    origin = key_position([mount_width / 2, -(mount_height / 2), 0], 1, corner)

    for i in range(len(origin)):
        origin[i] = origin[i] + thumb_offsets[i]

    if thumb_style == 'MINIDOX':
        origin[1] = origin[1] - .4*(minidox_Usize-1)*sa_length

    return origin


def default_thumb_tl_place(shape):
    logging.debug("thumb_tl_place()")
    shape = geometry_engine.rotate(shape, [7.5, -18, 10])
    shape = geometry_engine.translate(shape, thumborigin())
    shape = geometry_engine.translate(shape, [-32.5, -14.5, -2.5])
    return shape


def default_thumb_tr_place(shape):
    logging.debug("thumb_tr_place()")
    shape = geometry_engine.rotate(shape, [10, -15, 10])
    shape = geometry_engine.translate(shape, thumborigin())
    shape = geometry_engine.translate(shape, [-12, -16, 3])
    return shape


def default_thumb_mr_place(shape):
    logging.debug("thumb_mr_place()")
    shape = geometry_engine.rotate(shape, [-6, -34, 48])
    shape = geometry_engine.translate(shape, thumborigin())
    shape = geometry_engine.translate(shape, [-29, -40, -13])
    return shape


def default_thumb_ml_place(shape):
    logging.debug("thumb_ml_place()")
    shape = geometry_engine.rotate(shape, [6, -34, 40])
    shape = geometry_engine.translate(shape, thumborigin())
    shape = geometry_engine.translate(shape, [-51, -25, -12])
    return shape


def default_thumb_br_place(shape):
    logging.debug("thumb_br_place()")
    shape = geometry_engine.rotate(shape, [-16, -33, 54])
    shape = geometry_engine.translate(shape, thumborigin())
    shape = geometry_engine.translate(shape, [-37.8, -55.3, -25.3])
    return shape


def default_thumb_bl_place(shape):
    logging.debug("thumb_bl_place()")
    shape = geometry_engine.rotate(shape, [-4, -35, 52])
    shape = geometry_engine.translate(shape, thumborigin())
    shape = geometry_engine.translate(shape, [-56.3, -43.3, -23.5])
    return shape


def default_thumb_1x_layout(shape, cap=False):
    logging.debug("thumb_1x_layout()")
    if cap:
        shape_list = [
            default_thumb_mr_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_mr_rotation])),
            default_thumb_ml_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_ml_rotation])),
            default_thumb_br_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_br_rotation])),
            default_thumb_bl_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_bl_rotation])),
        ]

        if default_1U_cluster:
            shape_list.append(default_thumb_tr_place(geometry_engine.rotate(geometry_engine.rotate(shape, (0, 0, 90)), [0, 0, thumb_plate_tr_rotation])))
            shape_list.append(default_thumb_tr_place(geometry_engine.rotate(geometry_engine.rotate(shape, (0, 0, 90)), [0, 0, thumb_plate_tr_rotation])))
            shape_list.append(default_thumb_tl_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_tl_rotation])))
        shapes = add(shape_list)

    else:
        shape_list = [
            default_thumb_mr_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_mr_rotation])),
            default_thumb_ml_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_ml_rotation])),
            default_thumb_br_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_br_rotation])),
            default_thumb_bl_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_bl_rotation])),
        ]
        if default_1U_cluster:
            shape_list.append(default_thumb_tr_place(geometry_engine.rotate(geometry_engine.rotate(shape, (0, 0, 90)), [0, 0, thumb_plate_tr_rotation])))
        shapes = geometry_engine.union(shape_list)
    return shapes


def default_thumb_pcb_plate_cutouts(side="right"):
    shape = default_thumb_1x_layout(plate_pcb_cutout(side=side))
    shape = geometry_engine.union([shape, default_thumb_15x_layout(plate_pcb_cutout(side=side))])
    return shape


def default_thumb_15x_layout(shape, cap=False, plate=True):
    logging.debug("thumb_15x_layout()")
    if plate:
        if cap:
            shape = geometry_engine.rotate(shape, (0, 0, 90))
            cap_list = [default_thumb_tl_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_tl_rotation]))]
            cap_list.append(default_thumb_tr_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_tr_rotation])))
            return add(cap_list)
        else:
            shape_list = [default_thumb_tl_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_tl_rotation]))]
            if not default_1U_cluster:
                shape_list.append(default_thumb_tr_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_tr_rotation])))
            return geometry_engine.union(shape_list)
    else:
        if cap:
            shape = geometry_engine.rotate(shape, (0, 0, 90))
            shape_list = [
                default_thumb_tl_place(shape),
            ]
            shape_list.append(default_thumb_tr_place(shape))

            return add(shape_list)
        else:
            shape_list = [
                default_thumb_tl_place(shape),
            ]
            if not default_1U_cluster:
                shape_list.append(default_thumb_tr_place(shape))

            return geometry_engine.union(shape_list)


def adjustable_plate_size(Usize=1.5):
    return (Usize * sa_length - mount_height) / 2


def usize_dimention(Usize=1.5):
    return Usize * sa_length


def adjustable_plate_half(Usize=1.5):
    logging.debug("double_plate()")
    adjustable_plate_height = adjustable_plate_size(Usize)
    top_plate = geometry_engine.box(mount_width, adjustable_plate_height, web_thickness)
    top_plate = geometry_engine.translate(top_plate,
                                          [0, (adjustable_plate_height + mount_height) / 2, plate_thickness - (web_thickness / 2)]
                                          )
    return top_plate


def adjustable_plate(Usize=1.5):
    logging.debug("double_plate()")
    top_plate = adjustable_plate_half(Usize)
    return geometry_engine.union((top_plate, geometry_engine.mirror(top_plate, 'XZ')))


def adjustable_square_plate(Uwidth=1.5, Uheight=1.5):
    width = usize_dimention(Usize=Uwidth)
    height = usize_dimention(Usize=Uheight)
    print("width: {}, height: {}, thickness:{}".format(width, height, web_thickness))
    shape = geometry_engine.box(width, height, web_thickness)
    shape = geometry_engine.difference(shape, [geometry_engine.box(mount_width-.01, mount_height-.01, 2*web_thickness)])
    shape = geometry_engine.translate(shape, (0, 0, web_thickness/2))
    return shape


def double_plate_half():
    logging.debug("double_plate()")
    top_plate = geometry_engine.box(mount_width, double_plate_height, web_thickness)
    top_plate = geometry_engine.translate(top_plate,
                                          [0, (double_plate_height + mount_height) / 2, plate_thickness - (web_thickness / 2)]
                                          )
    return top_plate


def double_plate():
    logging.debug("double_plate()")
    top_plate = double_plate_half()
    return geometry_engine.union((top_plate, geometry_engine.mirror(top_plate, 'XZ')))


def thumbcaps(side='right', style_override=None):
    if style_override is None:
        _thumb_style = thumb_style
    else:
        _thumb_style = style_override

    if _thumb_style == "MINI":
        return mini_thumbcaps()
    elif _thumb_style == "MINIDOX":
        return minidox_thumbcaps()
    elif _thumb_style == "CARBONFET":
        return carbonfet_thumbcaps()

    elif "TRACKBALL" in _thumb_style:
        if (side == ball_side or ball_side == 'both'):
            if _thumb_style == "TRACKBALL_ORBYL":
                return tbjs_thumbcaps()
            elif _thumb_style == "TRACKBALL_CJ":
                return tbcj_thumbcaps()
        else:
            return thumbcaps(side, style_override=other_thumb)

    else:
        return default_thumbcaps()


def thumb(side="right", style_override=None):
    if style_override is None:
        _thumb_style = thumb_style
    else:
        _thumb_style = style_override

    if _thumb_style == "MINI":
        return mini_thumb(side)
    elif _thumb_style == "MINIDOX":
        return minidox_thumb(side)
    elif _thumb_style == "CARBONFET":
        return carbonfet_thumb(side)

    elif "TRACKBALL" in _thumb_style:
        if (side == ball_side or ball_side == 'both'):
            if _thumb_style == "TRACKBALL_ORBYL":
                return tbjs_thumb(side)
            elif _thumb_style == "TRACKBALL_CJ":
                return tbcj_thumb(side)
        else:
            return thumb(side, style_override=other_thumb)

    else:
        return default_thumb(side)


def thumb_connectors(side='right', style_override=None):
    if style_override is None:
        _thumb_style = thumb_style
    else:
        _thumb_style = style_override

    if _thumb_style == "MINI":
        return mini_thumb_connectors()
    elif _thumb_style == "MINIDOX":
        return minidox_thumb_connectors()
    elif _thumb_style == "CARBONFET":
        return carbonfet_thumb_connectors()

    elif "TRACKBALL" in _thumb_style:
        if (side == ball_side or ball_side == 'both'):
            if _thumb_style == "TRACKBALL_ORBYL":
                return tbjs_thumb_connectors()
            elif _thumb_style == "TRACKBALL_CJ":
                return tbcj_thumb_connectors()
        else:
            return thumb_connectors(side, style_override=other_thumb)

    else:
        return default_thumb_connectors()


def thumb_pcb_plate_cutouts(side='right', style_override=None):
    if style_override is None:
        _thumb_style = thumb_style
    else:
        _thumb_style = style_override

    if _thumb_style == "MINI":
        return mini_thumb_pcb_plate_cutouts(side)
    elif _thumb_style == "MINIDOX":
        return minidox_thumb_pcb_plate_cutouts(side)
    elif _thumb_style == "CARBONFET":
        return carbonfet_thumb_pcb_plate_cutouts(side)

    elif "TRACKBALL" in _thumb_style:
        if (side == ball_side or ball_side == 'both'):
            if _thumb_style == "TRACKBALL_ORBYL":
                return tbjs_thumb_pcb_plate_cutouts(side)
            elif _thumb_style == "TRACKBALL_CJ":
                return tbcj_thumb_pcb_plate_cutouts(side)
        else:
            return thumb_pcb_plate_cutouts(side, style_override=other_thumb)

    else:
        return default_thumb_pcb_plate_cutouts(side)


def default_thumbcaps():
    t1 = default_thumb_1x_layout(keycap(1), cap=True)
    if not default_1U_cluster:
        t1.add(default_thumb_15x_layout(keycap(1.5), cap=True))
    return t1


def default_thumb(side="right"):
    logging.debug("thumb()")
    shape = default_thumb_1x_layout(geometry_engine.rotate(single_plate(side=side), (0, 0, -90)))
    shape = geometry_engine.union([shape, default_thumb_15x_layout(geometry_engine.rotate(single_plate(side=side), (0, 0, -90)))])
    shape = geometry_engine.union([shape, default_thumb_15x_layout(double_plate(), plate=False)])
    #shape = add([shape, default_thumb_15x_layout(geometry_engine.rotate(single_plate(side=side), (0, 0, -90)))])
    #shape = add([shape, default_thumb_15x_layout(double_plate(), plate=False)])
    # if plate_pcb_clear:
    #     shape = geometry_engine.difference(shape, [default_thumb_pcb_plate_cutouts()])
    return shape


def thumb_post_tr():
    logging.debug("thumb_post_tr()")
    return geometry_engine.translate(web_post(),
                                     [(mount_width / 2) - post_adj, ((mount_height/2) + double_plate_height) - post_adj, 0]
                                     )


def thumb_post_tl():
    logging.debug("thumb_post_tl()")
    return geometry_engine.translate(web_post(),
                                     [-(mount_width / 2) + post_adj, ((mount_height/2) + double_plate_height) - post_adj, 0]
                                     )


def thumb_post_bl():
    logging.debug("thumb_post_bl()")
    return geometry_engine.translate(web_post(),
                                     [-(mount_width / 2) + post_adj, -((mount_height/2) + double_plate_height) + post_adj, 0]
                                     )


def thumb_post_br():
    logging.debug("thumb_post_br()")
    return geometry_engine.translate(web_post(),
                                     [(mount_width / 2) - post_adj, -((mount_height/2) + double_plate_height) + post_adj, 0]
                                     )


def default_thumb_connectors():
    logging.debug('thumb_connectors()')
    hulls = []

    # Top two
    if default_1U_cluster:
        hulls.append(
            triangle_hulls(
                [
                    default_thumb_tl_place(thumb_post_tr()),
                    default_thumb_tl_place(thumb_post_br()),
                    default_thumb_tr_place(web_post_tl()),
                    default_thumb_tr_place(web_post_bl()),
                ]
            )
        )
    else:
        hulls.append(
            triangle_hulls(
                [
                    default_thumb_tl_place(thumb_post_tr()),
                    default_thumb_tl_place(thumb_post_br()),
                    default_thumb_tr_place(thumb_post_tl()),
                    default_thumb_tr_place(thumb_post_bl()),
                ]
            )
        )

    # bottom two on the right
    hulls.append(
        triangle_hulls(
            [
                default_thumb_br_place(web_post_tr()),
                default_thumb_br_place(web_post_br()),
                default_thumb_mr_place(web_post_tl()),
                default_thumb_mr_place(web_post_bl()),
            ]
        )
    )

    # bottom two on the left
    hulls.append(
        triangle_hulls(
            [
                default_thumb_br_place(web_post_tr()),
                default_thumb_br_place(web_post_br()),
                default_thumb_mr_place(web_post_tl()),
                default_thumb_mr_place(web_post_bl()),
            ]
        )
    )
    # centers of the bottom four
    hulls.append(
        triangle_hulls(
            [
                default_thumb_bl_place(web_post_tr()),
                default_thumb_bl_place(web_post_br()),
                default_thumb_ml_place(web_post_tl()),
                default_thumb_ml_place(web_post_bl()),
            ]
        )
    )

    # top two to the middle two, starting on the left
    hulls.append(
        triangle_hulls(
            [
                default_thumb_br_place(web_post_tl()),
                default_thumb_bl_place(web_post_bl()),
                default_thumb_br_place(web_post_tr()),
                default_thumb_bl_place(web_post_br()),
                default_thumb_mr_place(web_post_tl()),
                default_thumb_ml_place(web_post_bl()),
                default_thumb_mr_place(web_post_tr()),
                default_thumb_ml_place(web_post_br()),
            ]
        )
    )

    if default_1U_cluster:
        hulls.append(
            triangle_hulls(
                [
                    default_thumb_tl_place(thumb_post_tl()),
                    default_thumb_ml_place(web_post_tr()),
                    default_thumb_tl_place(thumb_post_bl()),
                    default_thumb_ml_place(web_post_br()),
                    default_thumb_tl_place(thumb_post_br()),
                    default_thumb_mr_place(web_post_tr()),
                    default_thumb_tr_place(web_post_bl()),
                    default_thumb_mr_place(web_post_br()),
                    default_thumb_tr_place(web_post_br()),
                ]
            )
        )
    else:
        # top two to the main keyboard, starting on the left
        hulls.append(
            triangle_hulls(
                [
                    default_thumb_tl_place(thumb_post_tl()),
                    default_thumb_ml_place(web_post_tr()),
                    default_thumb_tl_place(thumb_post_bl()),
                    default_thumb_ml_place(web_post_br()),
                    default_thumb_tl_place(thumb_post_br()),
                    default_thumb_mr_place(web_post_tr()),
                    default_thumb_tr_place(thumb_post_bl()),
                    default_thumb_mr_place(web_post_br()),
                    default_thumb_tr_place(thumb_post_br()),
                ]
            )
        )

    if default_1U_cluster:
        hulls.append(
            triangle_hulls(
                [
                    default_thumb_tl_place(thumb_post_tl()),
                    key_place(web_post_bl(), 0, cornerrow),
                    default_thumb_tl_place(thumb_post_tr()),
                    key_place(web_post_br(), 0, cornerrow),
                    default_thumb_tr_place(web_post_tl()),
                    key_place(web_post_bl(), 1, cornerrow),
                    default_thumb_tr_place(web_post_tr()),
                    key_place(web_post_br(), 1, cornerrow),
                    key_place(web_post_bl(), 2, lastrow),
                    default_thumb_tr_place(web_post_tr()),
                    key_place(web_post_bl(), 2, lastrow),
                    default_thumb_tr_place(web_post_br()),
                    key_place(web_post_br(), 2, lastrow),
                    key_place(web_post_bl(), 3, lastrow),
                ]
            )
        )
    else:
        hulls.append(
            triangle_hulls(
                [
                    default_thumb_tl_place(thumb_post_tl()),
                    key_place(web_post_bl(), 0, cornerrow),
                    default_thumb_tl_place(thumb_post_tr()),
                    key_place(web_post_br(), 0, cornerrow),
                    default_thumb_tr_place(thumb_post_tl()),
                    key_place(web_post_bl(), 1, cornerrow),
                    default_thumb_tr_place(thumb_post_tr()),
                    key_place(web_post_br(), 1, cornerrow),
                    key_place(web_post_tl(), 2, lastrow),
                    key_place(web_post_bl(), 2, lastrow),
                    default_thumb_tr_place(thumb_post_tr()),
                    key_place(web_post_bl(), 2, lastrow),
                    default_thumb_tr_place(thumb_post_br()),
                    key_place(web_post_br(), 2, lastrow),
                    key_place(web_post_bl(), 3, lastrow),
                ]
            )
        )

    # return add(hulls)
    return geometry_engine.union(hulls)

############################
# MINI THUMB CLUSTER
############################


def mini_thumb_tr_place(shape):
    if mini_index_key:
        shape = geometry_engine.rotate(shape, [-25, 25, 0])
        shape = geometry_engine.translate(shape, thumborigin())
        shape = geometry_engine.translate(shape, [-12.5, -10, 2])
    else:
        shape = geometry_engine.rotate(shape, [14, -15, 10])
        shape = geometry_engine.translate(shape, thumborigin())
        shape = geometry_engine.translate(shape, [-15, -10, 5])
    return shape


def mini_thumb_tl_place(shape):
    shape = geometry_engine.rotate(shape, [10, -23, 25])
    shape = geometry_engine.translate(shape, thumborigin())
    shape = geometry_engine.translate(shape, [-35, -16, -2])
    return shape


def mini_thumb_mr_place(shape):
    shape = geometry_engine.rotate(shape, [10, -23, 25])
    shape = geometry_engine.translate(shape, thumborigin())
    shape = geometry_engine.translate(shape, [-23, -34, -6])
    return shape


def mini_thumb_br_place(shape):
    shape = geometry_engine.rotate(shape, [6, -34, 35])
    shape = geometry_engine.translate(shape, thumborigin())
    shape = geometry_engine.translate(shape, [-39, -43, -16])
    return shape


def mini_thumb_bl_place(shape):
    shape = geometry_engine.rotate(shape, [6, -32, 35])
    shape = geometry_engine.translate(shape, thumborigin())
    shape = geometry_engine.translate(shape, [-51, -25, -11.5])
    return shape


def mini_thumb_1x_layout(shape):
    return geometry_engine.union([
        # return add([
        mini_thumb_mr_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_mr_rotation])),
        mini_thumb_br_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_br_rotation])),
        mini_thumb_tl_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_tl_rotation])),
        mini_thumb_bl_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_bl_rotation])),
    ])


def mini_thumb_15x_layout(shape):
    return geometry_engine.union([mini_thumb_tr_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_tr_rotation]))])
    # return add([mini_thumb_tr_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_tr_rotation]))])


def mini_thumbcaps():
    t1 = mini_thumb_1x_layout(keycap(1))
    t15 = mini_thumb_15x_layout(geometry_engine.rotate(keycap(1), [0, 0, 90]))
    return t1.add(t15)


def mini_thumb(side="right"):
    shape = mini_thumb_1x_layout(single_plate(side=side))
    shape = geometry_engine.union([shape, mini_thumb_15x_layout(single_plate(side=side))])
    #shape = add([shape, mini_thumb_15x_layout(single_plate(side=side))])

    return shape


def mini_thumb_pcb_plate_cutouts(side="right"):
    shape = mini_thumb_1x_layout(plate_pcb_cutout(side=side))
    shape = geometry_engine.union([shape, mini_thumb_15x_layout(plate_pcb_cutout(side=side))])
    #shape = add([shape, mini_thumb_15x_layout(plate_pcb_cutout(side=side))])
    return shape


def mini_thumb_post_tr():
    return geometry_engine.translate(web_post(),
                                     [(mount_width / 2) - post_adj, (mount_height / 2) - post_adj, 0]
                                     )


def mini_thumb_post_tl():
    return geometry_engine.translate(web_post(),
                                     [-(mount_width / 2) + post_adj, (mount_height / 2) - post_adj, 0]
                                     )


def mini_thumb_post_bl():
    return geometry_engine.translate(web_post(),
                                     [-(mount_width / 2) + post_adj, -(mount_height / 2) + post_adj, 0]
                                     )


def mini_thumb_post_br():
    return geometry_engine.translate(web_post(),
                                     [(mount_width / 2) - post_adj, -(mount_height / 2) + post_adj, 0]
                                     )


def mini_thumb_connectors():
    hulls = []

    # Top two
    hulls.append(
        triangle_hulls(
            [
                mini_thumb_tl_place(web_post_tr()),
                mini_thumb_tl_place(web_post_br()),
                mini_thumb_tr_place(mini_thumb_post_tl()),
                mini_thumb_tr_place(mini_thumb_post_bl()),
            ]
        )
    )

    # bottom two on the right
    hulls.append(
        triangle_hulls(
            [
                mini_thumb_br_place(web_post_tr()),
                mini_thumb_br_place(web_post_br()),
                mini_thumb_mr_place(web_post_tl()),
                mini_thumb_mr_place(web_post_bl()),
            ]
        )
    )

    # bottom two on the left
    hulls.append(
        triangle_hulls(
            [
                mini_thumb_mr_place(web_post_tr()),
                mini_thumb_mr_place(web_post_br()),
                mini_thumb_tr_place(mini_thumb_post_br()),
            ]
        )
    )

    # between top and bottom row
    hulls.append(
        triangle_hulls(
            [
                mini_thumb_br_place(web_post_tl()),
                mini_thumb_bl_place(web_post_bl()),
                mini_thumb_br_place(web_post_tr()),
                mini_thumb_bl_place(web_post_br()),
                mini_thumb_mr_place(web_post_tl()),
                mini_thumb_tl_place(web_post_bl()),
                mini_thumb_mr_place(web_post_tr()),
                mini_thumb_tl_place(web_post_br()),
                mini_thumb_tr_place(web_post_bl()),
                mini_thumb_mr_place(web_post_tr()),
                mini_thumb_tr_place(web_post_br()),
            ]
        )
    )
    # top two to the main keyboard, starting on the left
    hulls.append(
        triangle_hulls(
            [
                mini_thumb_tl_place(web_post_tl()),
                mini_thumb_bl_place(web_post_tr()),
                mini_thumb_tl_place(web_post_bl()),
                mini_thumb_bl_place(web_post_br()),
                mini_thumb_mr_place(web_post_tr()),
                mini_thumb_tl_place(web_post_bl()),
                mini_thumb_tl_place(web_post_br()),
                mini_thumb_mr_place(web_post_tr()),
            ]
        )
    )
    # top two to the main keyboard, starting on the left
    hulls.append(
        triangle_hulls(
            [
                mini_thumb_tl_place(web_post_tl()),
                key_place(web_post_bl(), 0, cornerrow),
                mini_thumb_tl_place(web_post_tr()),
                key_place(web_post_br(), 0, cornerrow),
                mini_thumb_tr_place(mini_thumb_post_tl()),
                key_place(web_post_bl(), 1, cornerrow),
                mini_thumb_tr_place(mini_thumb_post_tr()),
                key_place(web_post_br(), 1, cornerrow),
                # key_place(web_post_tl(), 2, lastrow),
                key_place(web_post_bl(), 2, lastrow),
                mini_thumb_tr_place(mini_thumb_post_tr()),
                key_place(web_post_bl(), 2, lastrow),
                mini_thumb_tr_place(mini_thumb_post_br()),
                key_place(web_post_br(), 2, lastrow),
                key_place(web_post_bl(), 3, lastrow),
            ]
        )
    )

    return geometry_engine.union(hulls)
    # return add(hulls)


############################
# MINIDOX (3-key) THUMB CLUSTER
############################

def minidox_thumb_tl_place(shape):
    shape = geometry_engine.rotate(shape, [10, -23, 25])
    shape = geometry_engine.translate(shape, thumborigin())
    shape = geometry_engine.translate(shape, [-35, -16, -2])
    return shape


def minidox_thumb_tr_place(shape):
    shape = geometry_engine.rotate(shape, [14, -15, 10])
    shape = geometry_engine.translate(shape, thumborigin())
    shape = geometry_engine.translate(shape, [-15, -10, 5])
    return shape


def minidox_thumb_ml_place(shape):
    shape = geometry_engine.rotate(shape, [6, -34, 40])
    shape = geometry_engine.translate(shape, thumborigin())
    shape = geometry_engine.translate(shape, [-53, -26, -12])
    return shape


def minidox_thumb_1x_layout(shape):
    return geometry_engine.union([
        # return add([
        minidox_thumb_tr_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_tr_rotation])),
        minidox_thumb_tl_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_tl_rotation])),
        minidox_thumb_ml_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_ml_rotation])),
    ])


def minidox_thumb_fx_layout(shape):
    return geometry_engine.union([
        # return add([
        minidox_thumb_tr_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_tr_rotation])),
        minidox_thumb_tl_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_tl_rotation])),
        minidox_thumb_ml_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_ml_rotation])),
    ])


def minidox_thumbcaps():
    t1 = minidox_thumb_1x_layout(keycap(1))
    return t1


def minidox_thumb(side="right"):

    shape = minidox_thumb_fx_layout(geometry_engine.rotate(single_plate(side=side), [0.0, 0.0, -90]))
    shape = geometry_engine.union([shape, minidox_thumb_fx_layout(adjustable_plate(minidox_Usize))])
    #shape = add([shape, minidox_thumb_fx_layout(adjustable_plate(minidox_Usize))])
    # shape = minidox_thumb_1x_layout(single_plate(side=side))
    return shape


def minidox_thumb_pcb_plate_cutouts(side="right"):
    shape = minidox_thumb_fx_layout(plate_pcb_cutout(side=side))
    shape = geometry_engine.union([shape, minidox_thumb_fx_layout(plate_pcb_cutout())])
    #shape = add([shape, minidox_thumb_fx_layout(plate_pcb_cutout())])
    return shape


def minidox_thumb_post_tr():
    logging.debug("thumb_post_tr()")
    return geometry_engine.translate(web_post(),
                                     [(mount_width / 2) - post_adj, ((mount_height/2) + adjustable_plate_size(minidox_Usize)) - post_adj, 0]
                                     )


def minidox_thumb_post_tl():
    logging.debug("thumb_post_tl()")
    return geometry_engine.translate(web_post(),
                                     [-(mount_width / 2) + post_adj, ((mount_height/2) + adjustable_plate_size(minidox_Usize)) - post_adj, 0]
                                     )


def minidox_thumb_post_bl():
    logging.debug("thumb_post_bl()")
    return geometry_engine.translate(web_post(),
                                     [-(mount_width / 2) + post_adj, -((mount_height/2) + adjustable_plate_size(minidox_Usize)) + post_adj, 0]
                                     )


def minidox_thumb_post_br():
    logging.debug("thumb_post_br()")
    return geometry_engine.translate(web_post(),
                                     [(mount_width / 2) - post_adj, -((mount_height/2) + adjustable_plate_size(minidox_Usize)) + post_adj, 0]
                                     )


def minidox_thumb_connectors():
    hulls = []

    # Top two
    hulls.append(
        triangle_hulls(
            [
                minidox_thumb_tl_place(minidox_thumb_post_tr()),
                minidox_thumb_tl_place(minidox_thumb_post_br()),
                minidox_thumb_tr_place(minidox_thumb_post_tl()),
                minidox_thumb_tr_place(minidox_thumb_post_bl()),
            ]
        )
    )

    # bottom two on the right
    hulls.append(
        triangle_hulls(
            [
                minidox_thumb_tl_place(minidox_thumb_post_tl()),
                minidox_thumb_tl_place(minidox_thumb_post_bl()),
                minidox_thumb_ml_place(minidox_thumb_post_tr()),
                minidox_thumb_ml_place(minidox_thumb_post_br()),
            ]
        )
    )

    # top two to the main keyboard, starting on the left
    hulls.append(
        triangle_hulls(
            [
                minidox_thumb_tl_place(minidox_thumb_post_tl()),
                key_place(web_post_bl(), 0, cornerrow),
                minidox_thumb_tl_place(minidox_thumb_post_tr()),
                key_place(web_post_br(), 0, cornerrow),
                minidox_thumb_tr_place(minidox_thumb_post_tl()),
                key_place(web_post_bl(), 1, cornerrow),
                minidox_thumb_tr_place(minidox_thumb_post_tr()),
                key_place(web_post_br(), 1, cornerrow),
                key_place(web_post_bl(), 2, lastrow),
                minidox_thumb_tr_place(minidox_thumb_post_tr()),
                key_place(web_post_bl(), 2, lastrow),
                minidox_thumb_tr_place(minidox_thumb_post_br()),
                key_place(web_post_br(), 2, lastrow),
                key_place(web_post_bl(), 3, lastrow),
            ]
        )
    )

    return geometry_engine.union(hulls)
    # return add(hulls)


############################
# Carbonfet THUMB CLUSTER
############################


def carbonfet_thumb_tl_place(shape):
    shape = geometry_engine.rotate(shape, [10, -24, 10])
    shape = geometry_engine.translate(shape, thumborigin())
    shape = geometry_engine.translate(shape, [-13, -9.8, 4])
    return shape


def carbonfet_thumb_tr_place(shape):
    shape = geometry_engine.rotate(shape, [6, -25, 10])
    shape = geometry_engine.translate(shape, thumborigin())
    shape = geometry_engine.translate(shape, [-7.5, -29.5, 0])
    return shape


def carbonfet_thumb_ml_place(shape):
    shape = geometry_engine.rotate(shape, [8, -31, 14])
    shape = geometry_engine.translate(shape, thumborigin())
    shape = geometry_engine.translate(shape, [-30.5, -17, -6])
    return shape


def carbonfet_thumb_mr_place(shape):
    shape = geometry_engine.rotate(shape, [4, -31, 14])
    shape = geometry_engine.translate(shape, thumborigin())
    shape = geometry_engine.translate(shape, [-22.2, -41, -10.3])
    return shape


def carbonfet_thumb_br_place(shape):
    shape = geometry_engine.rotate(shape, [2, -37, 18])
    shape = geometry_engine.translate(shape, thumborigin())
    shape = geometry_engine.translate(shape, [-37, -46.4, -22])
    return shape


def carbonfet_thumb_bl_place(shape):
    shape = geometry_engine.rotate(shape, [6, -37, 18])
    shape = geometry_engine.translate(shape, thumborigin())
    shape = geometry_engine.translate(shape, [-47, -23, -19])
    return shape


def carbonfet_thumb_1x_layout(shape):
    return geometry_engine.union([
        # return add([
        carbonfet_thumb_tr_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_tr_rotation])),
        carbonfet_thumb_mr_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_mr_rotation])),
        carbonfet_thumb_br_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_br_rotation])),
        carbonfet_thumb_tl_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_tl_rotation])),
    ])


def carbonfet_thumb_15x_layout(shape, plate=True):
    if plate:
        return geometry_engine.union([
            # return add([
            carbonfet_thumb_bl_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_bl_rotation])),
            carbonfet_thumb_ml_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_ml_rotation]))
        ])
    else:
        return geometry_engine.union([
            # return add([
            carbonfet_thumb_bl_place(shape),
            carbonfet_thumb_ml_place(shape)
        ])


def carbonfet_thumbcaps():
    t1 = carbonfet_thumb_1x_layout(keycap(1))
    t15 = carbonfet_thumb_15x_layout(geometry_engine.rotate(keycap(1.5), [0, 0, 90]))
    return t1.add(t15)


def carbonfet_thumb(side="right"):
    shape = carbonfet_thumb_1x_layout(single_plate(side=side))
    shape = geometry_engine.union([shape, carbonfet_thumb_15x_layout(double_plate_half(), plate=False)])
    shape = geometry_engine.union([shape, carbonfet_thumb_15x_layout(single_plate(side=side))])
    #shape = add([shape, carbonfet_thumb_15x_layout(double_plate_half(), plate=False)])
    #shape = add([shape, carbonfet_thumb_15x_layout(single_plate(side=side))])

    return shape


def carbonfet_thumb_pcb_plate_cutouts(side="right"):
    shape = carbonfet_thumb_1x_layout(plate_pcb_cutout(side=side))
    shape = geometry_engine.union([shape, carbonfet_thumb_15x_layout(plate_pcb_cutout())])
    #shape = add([shape, carbonfet_thumb_15x_layout(plate_pcb_cutout())])
    return shape


def carbonfet_thumb_post_tr():
    return geometry_engine.translate(web_post(),
                                     [(mount_width / 2) - post_adj, (mount_height / 1.15) - post_adj, 0]
                                     )


def carbonfet_thumb_post_tl():
    return geometry_engine.translate(web_post(),
                                     [-(mount_width / 2) + post_adj, (mount_height / 1.15) - post_adj, 0]
                                     )


def carbonfet_thumb_post_bl():
    return geometry_engine.translate(web_post(),
                                     [-(mount_width / 2) + post_adj, -(mount_height / 1.15) + post_adj, 0]
                                     )


def carbonfet_thumb_post_br():
    return geometry_engine.translate(web_post(),
                                     [(mount_width / 2) - post_adj, -(mount_height / 2) + post_adj, 0]
                                     )


def carbonfet_thumb_connectors():
    hulls = []

    # Top two
    hulls.append(
        triangle_hulls(
            [
                carbonfet_thumb_tl_place(web_post_tl()),
                carbonfet_thumb_tl_place(web_post_bl()),
                carbonfet_thumb_ml_place(carbonfet_thumb_post_tr()),
                carbonfet_thumb_ml_place(web_post_br()),
            ]
        )
    )

    hulls.append(
        triangle_hulls(
            [
                carbonfet_thumb_ml_place(carbonfet_thumb_post_tl()),
                carbonfet_thumb_ml_place(web_post_bl()),
                carbonfet_thumb_bl_place(carbonfet_thumb_post_tr()),
                carbonfet_thumb_bl_place(web_post_br()),
            ]
        )
    )

    # bottom two on the right
    hulls.append(
        triangle_hulls(
            [
                carbonfet_thumb_br_place(web_post_tr()),
                carbonfet_thumb_br_place(web_post_br()),
                carbonfet_thumb_mr_place(web_post_tl()),
                carbonfet_thumb_mr_place(web_post_bl()),
            ]
        )
    )

    # bottom two on the left
    hulls.append(
        triangle_hulls(
            [
                carbonfet_thumb_mr_place(web_post_tr()),
                carbonfet_thumb_mr_place(web_post_br()),
                carbonfet_thumb_tr_place(web_post_tl()),
                carbonfet_thumb_tr_place(web_post_bl()),
            ]
        )
    )
    hulls.append(
        triangle_hulls(
            [
                carbonfet_thumb_tr_place(web_post_br()),
                carbonfet_thumb_tr_place(web_post_bl()),
                carbonfet_thumb_mr_place(web_post_br()),
            ]
        )
    )

    # between top and bottom row
    hulls.append(
        triangle_hulls(
            [
                carbonfet_thumb_br_place(web_post_tl()),
                carbonfet_thumb_bl_place(web_post_bl()),
                carbonfet_thumb_br_place(web_post_tr()),
                carbonfet_thumb_bl_place(web_post_br()),
                carbonfet_thumb_mr_place(web_post_tl()),
                carbonfet_thumb_ml_place(web_post_bl()),
                carbonfet_thumb_mr_place(web_post_tr()),
                carbonfet_thumb_ml_place(web_post_br()),
                carbonfet_thumb_tr_place(web_post_tl()),
                carbonfet_thumb_tl_place(web_post_bl()),
                carbonfet_thumb_tr_place(web_post_tr()),
                carbonfet_thumb_tl_place(web_post_br()),
            ]
        )
    )
    # top two to the main keyboard, starting on the left
    hulls.append(
        triangle_hulls(
            [
                carbonfet_thumb_ml_place(carbonfet_thumb_post_tl()),
                key_place(web_post_bl(), 0, cornerrow),
                carbonfet_thumb_ml_place(carbonfet_thumb_post_tr()),
                key_place(web_post_br(), 0, cornerrow),
                carbonfet_thumb_tl_place(web_post_tl()),
                key_place(web_post_bl(), 1, cornerrow),
                carbonfet_thumb_tl_place(web_post_tr()),
                key_place(web_post_br(), 1, cornerrow),
                key_place(web_post_bl(), 2, lastrow),
                carbonfet_thumb_tl_place(web_post_tr()),
                key_place(web_post_bl(), 2, lastrow),
                carbonfet_thumb_tl_place(web_post_br()),
                key_place(web_post_br(), 2, lastrow),
                key_place(web_post_bl(), 3, lastrow),
                carbonfet_thumb_tl_place(web_post_br()),
                carbonfet_thumb_tr_place(web_post_tr()),
            ]
        )
    )

    hulls.append(
        triangle_hulls(
            [
                carbonfet_thumb_tr_place(web_post_br()),
                carbonfet_thumb_tr_place(web_post_tr()),
                key_place(web_post_bl(), 3, lastrow),
            ]
        )
    )

    return geometry_engine.union(hulls)
    # return add(hulls)


############################
# Trackball (Ball + 4-key) THUMB CLUSTER
############################

def tbjs_thumb_position_rotation():
    rot = [10, -15, 5]
    pos = thumborigin()
    # Changes size based on key diameter around ball, shifting off of the top left cluster key.
    shift = [-.9*tbjs_key_diameter/2+27-42, -.1*tbjs_key_diameter/2+3-25, -5]
    for i in range(len(pos)):
        pos[i] = pos[i] + shift[i] + tbjs_translation_offset[i]

    for i in range(len(rot)):
        rot[i] = rot[i] + tbjs_rotation_offset[i]

    return pos, rot


def tbjs_place(shape):
    pos, rot = tbjs_thumb_position_rotation()
    shape = geometry_engine.rotate(shape, rot)
    shape = geometry_engine.translate(shape, pos)
    return shape


def tbjs_thumb_tl_place(shape):
    logging.debug("thumb_tr_place()")
    # Modifying to make a "ring" of keys
    shape = geometry_engine.rotate(shape, [0, 0, 0])
    t_off = tbjs_key_translation_offsets[0]
    shape = geometry_engine.rotate(shape, tbjs_key_rotation_offsets[0])
    shape = geometry_engine.translate(shape, (t_off[0], t_off[1]+tbjs_key_diameter/2, t_off[2]))
    shape = geometry_engine.rotate(shape, [0, 0, -80])
    shape = tbjs_place(shape)

    return shape


def tbjs_thumb_mr_place(shape):
    logging.debug("thumb_mr_place()")
    shape = geometry_engine.rotate(shape, [0, 0, 0])
    shape = geometry_engine.rotate(shape, tbjs_key_rotation_offsets[1])
    t_off = tbjs_key_translation_offsets[1]
    shape = geometry_engine.translate(shape, (t_off[0], t_off[1]+tbjs_key_diameter/2, t_off[2]))
    shape = geometry_engine.rotate(shape, [0, 0, -130])
    shape = tbjs_place(shape)

    return shape


def tbjs_thumb_br_place(shape):
    logging.debug("thumb_br_place()")

    shape = geometry_engine.rotate(shape, [0, 0, 180])
    shape = geometry_engine.rotate(shape, tbjs_key_rotation_offsets[2])
    t_off = tbjs_key_translation_offsets[2]
    shape = geometry_engine.translate(shape, (t_off[0], t_off[1]+tbjs_key_diameter/2, t_off[2]))
    shape = geometry_engine.rotate(shape, [0, 0, -180])
    shape = tbjs_place(shape)

    return shape


def tbjs_thumb_bl_place(shape):
    logging.debug("thumb_bl_place()")
    shape = geometry_engine.rotate(shape, [0, 0, 180])
    shape = geometry_engine.rotate(shape, tbjs_key_rotation_offsets[3])
    t_off = tbjs_key_translation_offsets[3]
    shape = geometry_engine.translate(shape, (t_off[0], t_off[1]+tbjs_key_diameter/2, t_off[2]))
    shape = geometry_engine.rotate(shape, [0, 0, -230])
    shape = tbjs_place(shape)

    return shape


def tbjs_thumb_1x_layout(shape):
    return geometry_engine.union([
        # return add([
        tbjs_thumb_tl_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_tr_rotation])),
        tbjs_thumb_mr_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_mr_rotation])),
        tbjs_thumb_bl_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_bl_rotation])),
        tbjs_thumb_br_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_br_rotation])),
    ])


def tbjs_thumb_pcb_plate_cutouts(side="right"):
    return tbjs_thumb_1x_layout(plate_pcb_cutout(side=side))


def tbjs_thumb_fx_layout(shape):
    return [
        tbjs_thumb_tl_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_tr_rotation])),
        tbjs_thumb_mr_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_mr_rotation])),
        tbjs_thumb_bl_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_bl_rotation])),
        tbjs_thumb_br_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_br_rotation])),
    ]


def trackball_layout(shape):
    return geometry_engine.union([
        # return add([
        tbjs_place(shape),
    ])


def tbjs_thumbcaps():
    t1 = tbjs_thumb_1x_layout(keycap(1))
    return t1


def tbjs_thumb(side="right"):
    # shape = tbjs_thumb_fx_layout(geometry_engine.rotate(single_plate(side=side), [0.0, 0.0, -90]))
    shape = tbjs_thumb_1x_layout(single_plate(side=side))
    # shape = tbjs_thumb_fx_layout(adjustable_square_plate(Uwidth=tbjs_Uwidth, Uheight=tbjs_Uheight))
    shape = geometry_engine.union([shape, *tbjs_thumb_fx_layout(adjustable_square_plate(Uwidth=tbjs_Uwidth, Uheight=tbjs_Uheight))])
    #shape = add([shape, *tbjs_thumb_fx_layout(adjustable_square_plate(Uwidth=tbjs_Uwidth, Uheight=tbjs_Uheight))])

    # shape = geometry_engine.union([shape, trackball_layout(trackball_socket())])
    # shape = tbjs_thumb_1x_layout(single_plate(side=side))
    return shape


def tbjs_thumb_post_tr():
    logging.debug("thumb_post_tr()")
    return geometry_engine.translate(web_post(),
                                     [(mount_width / 2) + adjustable_plate_size(tbjs_Uwidth) - post_adj,
                                      ((mount_height/2) + adjustable_plate_size(tbjs_Uheight)) - post_adj, 0]
                                     )


def tbjs_thumb_post_tl():
    logging.debug("thumb_post_tl()")
    return geometry_engine.translate(web_post(),
                                     [-(mount_width / 2) - adjustable_plate_size(tbjs_Uwidth) + post_adj,
                                      ((mount_height/2) + adjustable_plate_size(tbjs_Uheight)) - post_adj, 0]
                                     )


def tbjs_thumb_post_bl():
    logging.debug("thumb_post_bl()")
    return geometry_engine.translate(web_post(),
                                     [-(mount_width / 2) - adjustable_plate_size(tbjs_Uwidth) + post_adj, -
                                      ((mount_height/2) + adjustable_plate_size(tbjs_Uheight)) + post_adj, 0]
                                     )


def tbjs_thumb_post_br():
    logging.debug("thumb_post_br()")
    return geometry_engine.translate(web_post(),
                                     [(mount_width / 2) + adjustable_plate_size(tbjs_Uwidth) - post_adj, -
                                      ((mount_height/2) + adjustable_plate_size(tbjs_Uheight)) + post_adj, 0]
                                     )


def tbjs_post_r():
    logging.debug("tbjs_post_r()")
    radius = ball_diameter/2 + ball_wall_thickness + ball_gap
    return geometry_engine.translate(web_post(),
                                     [1.0*(radius - post_adj), 0.0*(radius - post_adj), 0]
                                     )


def tbjs_post_tr():
    logging.debug("tbjs_post_tr()")
    radius = ball_diameter/2+ball_wall_thickness + ball_gap
    return geometry_engine.translate(web_post(),
                                     [0.5*(radius - post_adj), 0.866*(radius - post_adj), 0]
                                     )


def tbjs_post_tl():
    logging.debug("tbjs_post_tl()")
    radius = ball_diameter/2+ball_wall_thickness + ball_gap
    return geometry_engine.translate(web_post(),
                                     [-0.5*(radius - post_adj), 0.866*(radius - post_adj), 0]
                                     )


def tbjs_post_l():
    logging.debug("tbjs_post_l()")
    radius = ball_diameter/2+ball_wall_thickness + ball_gap
    return geometry_engine.translate(web_post(),
                                     [-1.0*(radius - post_adj), 0.0*(radius - post_adj), 0]
                                     )


def tbjs_post_bl():
    logging.debug("tbjs_post_bl()")
    radius = ball_diameter/2+ball_wall_thickness + ball_gap
    return geometry_engine.translate(web_post(),
                                     [-0.5*(radius - post_adj), -0.866*(radius - post_adj), 0]
                                     )


def tbjs_post_br():
    logging.debug("tbjs_post_br()")
    radius = ball_diameter/2+ball_wall_thickness + ball_gap
    return geometry_engine.translate(web_post(),
                                     [0.5*(radius - post_adj), -0.866*(radius - post_adj), 0]
                                     )


def tbjs_thumb_connectors():
    logging.debug('thumb_connectors()')
    hulls = []

    # bottom 2 to tb
    hulls.append(
        triangle_hulls(
            [
                tbjs_place(tbjs_post_l()),
                tbjs_thumb_bl_place(tbjs_thumb_post_tl()),
                tbjs_place(tbjs_post_bl()),
                tbjs_thumb_bl_place(tbjs_thumb_post_tr()),
                tbjs_thumb_br_place(tbjs_thumb_post_tl()),
                tbjs_place(tbjs_post_bl()),
                tbjs_thumb_br_place(tbjs_thumb_post_tr()),
                tbjs_place(tbjs_post_br()),
                tbjs_thumb_br_place(tbjs_thumb_post_tr()),
                tbjs_place(tbjs_post_br()),
                tbjs_thumb_mr_place(tbjs_thumb_post_br()),
                tbjs_place(tbjs_post_r()),
                tbjs_thumb_mr_place(tbjs_thumb_post_bl()),
                tbjs_thumb_tl_place(tbjs_thumb_post_br()),
                tbjs_place(tbjs_post_r()),
                tbjs_thumb_tl_place(tbjs_thumb_post_bl()),
                tbjs_place(tbjs_post_tr()),
                key_place(web_post_bl(), 0, cornerrow),
                tbjs_place(tbjs_post_tl()),
            ]
        )
    )

    # bottom left
    hulls.append(
        triangle_hulls(
            [
                tbjs_thumb_bl_place(tbjs_thumb_post_tr()),
                tbjs_thumb_br_place(tbjs_thumb_post_tl()),
                tbjs_thumb_bl_place(tbjs_thumb_post_br()),
                tbjs_thumb_br_place(tbjs_thumb_post_bl()),
            ]
        )
    )

    # bottom right
    hulls.append(
        triangle_hulls(
            [
                tbjs_thumb_br_place(tbjs_thumb_post_tr()),
                tbjs_thumb_mr_place(tbjs_thumb_post_br()),
                tbjs_thumb_br_place(tbjs_thumb_post_br()),
                tbjs_thumb_mr_place(tbjs_thumb_post_tr()),
            ]
        )
    )
    # top right
    hulls.append(
        triangle_hulls(
            [
                tbjs_thumb_mr_place(tbjs_thumb_post_bl()),
                tbjs_thumb_tl_place(tbjs_thumb_post_br()),
                tbjs_thumb_mr_place(tbjs_thumb_post_tl()),
                tbjs_thumb_tl_place(tbjs_thumb_post_tr()),
            ]
        )
    )

    return geometry_engine.union(hulls)
    # return add(hulls)


############################
# TRACKBALL THUMB CLUSTER
############################

# single_plate = the switch shape

def tbcj_thumb_tr_place(shape):
    shape = geometry_engine.rotate(shape, [10, -15, 10])
    shape = geometry_engine.translate(shape, thumborigin())
    shape = geometry_engine.translate(shape, [-12, -16, 3])
    return shape


def tbcj_thumb_tl_place(shape):
    shape = geometry_engine.rotate(shape, [7.5, -18, 10])
    shape = geometry_engine.translate(shape, thumborigin())
    shape = geometry_engine.translate(shape, [-32.5, -14.5, -2.5])
    return shape


def tbcj_thumb_ml_place(shape):
    shape = geometry_engine.rotate(shape, [6, -34, 40])
    shape = geometry_engine.translate(shape, thumborigin())
    shape = geometry_engine.translate(shape, [-51, -25, -12])
    return shape


def tbcj_thumb_bl_place(shape):
    shape = geometry_engine.rotate(shape, [-4, -35, 52])
    shape = geometry_engine.translate(shape, thumborigin())
    shape = geometry_engine.translate(shape, [-56.3, -43.3, -23.5])
    return shape


def tbcj_thumb_layout(shape):
    return geometry_engine.union([
        # return add([
        tbcj_thumb_tr_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_tr_rotation])),
        tbcj_thumb_tl_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_tl_rotation])),
        tbcj_thumb_ml_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_ml_rotation])),
        tbcj_thumb_bl_place(geometry_engine.rotate(shape, [0, 0, thumb_plate_bl_rotation])),
    ])


# def oct_corner(i, radius, shape):
#    i = (i+1)%8
#
#    points_x = [1, 2, 2, 1, -1, -2, -2, -1]
#    points_y = [2, 1, -1, -2, -2, -1, 1, 2]
#
#    return geometry_engine.translate(shape, (points_x[i] * radius / 2, points_y[i] * radius / 2, 0))


def oct_corner(i, diameter, shape):
    radius = diameter / 2
    i = (i+1) % 8

    r = radius
    m = radius * math.tan(math.pi / 8)

    points_x = [m, r, r, m, -m, -r, -r, -m]
    points_y = [r, m, -m, -r, -r, -m, m, r]

    return geometry_engine.translate(shape, (points_x[i], points_y[i], 0))


def tbcj_edge_post(i):
    shape = geometry_engine.box(post_size, post_size, tbcj_thickness)
    shape = oct_corner(i, tbcj_outer_diameter, shape)
    return shape


def tbcj_web_post(i):
    shape = geometry_engine.box(post_size, post_size, tbcj_thickness)
    shape = oct_corner(i, tbcj_outer_diameter, shape)
    return shape


def tbcj_holder():
    center = geometry_engine.box(post_size, post_size, tbcj_thickness)

    shape = []
    for i in range(8):
        shape_ = geometry_engine.convex_hull([
            center,
            tbcj_edge_post(i),
            tbcj_edge_post(i + 1),
        ])
        shape.append(shape_)
    shape = geometry_engine.union(shape)

    shape = geometry_engine.difference(
        shape,
        [geometry_engine.cylinder(tbcj_inner_diameter/2, tbcj_thickness + 0.1)]
    )

    return shape


def tbcj_thumb_position_rotation():
    pos = np.array([-15, -60, -12]) + thumborigin()
    rot = (0, 0, 0)
    return pos, rot


def tbcj_place(shape):
    loc = np.array([-15, -60, -12]) + thumborigin()
    shape = geometry_engine.translate(shape, loc)
    shape = geometry_engine.rotate(shape, (0, 0, 0))
    return shape


def tbcj_thumb(side="right"):
    t = tbcj_thumb_layout(single_plate(side=side))
    tb = tbcj_place(tbcj_holder())
    return geometry_engine.union([t, tb])
    # return add([t, tb])


def tbcj_thumb_pcb_plate_cutouts(side="right"):
    t = tbcj_thumb_layout(plate_pcb_cutout(side=side))
    return t


def tbcj_thumbcaps():
    t = tbcj_thumb_layout(keycap(1))
    return t


# TODO:  VERIFY THEY CAN BE DELETED.  THEY LOOK LIKE REPLICATES.
# def thumb_post_tr():
#     return geometry_engine.translate(web_post(),
#                      [(mount_width / 2) - post_adj, ((mount_height/2) + double_plate_height) - post_adj, 0]
#                      )
#
#
# def thumb_post_tl():
#     return geometry_engine.translate(web_post(),
#                      [-(mount_width / 2) + post_adj, ((mount_height/2) + double_plate_height) - post_adj, 0]
#                      )
#
#
# def thumb_post_bl():
#     return geometry_engine.translate(web_post(),
#                      [-(mount_width / 2) + post_adj, -((mount_height/2) + double_plate_height) + post_adj, 0]
#                      )
#
#
# def thumb_post_br():
#     return geometry_engine.translate(web_post(),
#                      [(mount_width / 2) - post_adj, -((mount_height/2) + double_plate_height) + post_adj, 0]
#                      )

def tbcj_thumb_connectors():
    hulls = []

    # Top two
    hulls.append(
        triangle_hulls(
            [
                tbcj_thumb_tl_place(web_post_tr()),
                tbcj_thumb_tl_place(web_post_br()),
                tbcj_thumb_tr_place(web_post_tl()),
                tbcj_thumb_tr_place(web_post_bl()),
            ]
        )
    )

    # centers of the bottom four
    hulls.append(
        triangle_hulls(
            [
                tbcj_thumb_bl_place(web_post_tr()),
                tbcj_thumb_bl_place(web_post_br()),
                tbcj_thumb_ml_place(web_post_tl()),
                tbcj_thumb_ml_place(web_post_bl()),
            ]
        )
    )

    # top two to the middle two, starting on the left

    hulls.append(
        triangle_hulls(
            [
                tbcj_thumb_tl_place(web_post_tl()),
                tbcj_thumb_ml_place(web_post_tr()),
                tbcj_thumb_tl_place(web_post_bl()),
                tbcj_thumb_ml_place(web_post_br()),
                tbcj_thumb_tl_place(web_post_br()),
                tbcj_thumb_tr_place(web_post_bl()),
                tbcj_thumb_tr_place(web_post_br()),
            ]
        )
    )

    hulls.append(
        triangle_hulls(
            [
                tbcj_thumb_tl_place(web_post_tl()),
                key_place(web_post_bl(), 0, cornerrow),
                tbcj_thumb_tl_place(web_post_tr()),
                key_place(web_post_br(), 0, cornerrow),
                tbcj_thumb_tr_place(web_post_tl()),
                key_place(web_post_bl(), 1, cornerrow),
                tbcj_thumb_tr_place(web_post_tr()),
                key_place(web_post_br(), 1, cornerrow),
                key_place(web_post_bl(), 2, lastrow),
                tbcj_thumb_tr_place(web_post_tr()),
                key_place(web_post_bl(), 2, lastrow),
                tbcj_thumb_tr_place(web_post_br()),
                key_place(web_post_br(), 2, lastrow),
                key_place(web_post_bl(), 3, lastrow),
            ]
        )
    )

    hulls.append(
        triangle_hulls(
            [
                tbcj_place(tbcj_web_post(4)),
                tbcj_thumb_bl_place(web_post_bl()),
                tbcj_place(tbcj_web_post(5)),
                tbcj_thumb_bl_place(web_post_br()),
                tbcj_place(tbcj_web_post(6)),
            ]
        )
    )

    hulls.append(
        triangle_hulls(
            [
                tbcj_thumb_bl_place(web_post_br()),
                tbcj_place(tbcj_web_post(6)),
                tbcj_thumb_ml_place(web_post_bl()),
            ]
        )
    )

    hulls.append(
        triangle_hulls(
            [
                tbcj_thumb_ml_place(web_post_bl()),
                tbcj_place(tbcj_web_post(6)),
                tbcj_thumb_ml_place(web_post_br()),
                tbcj_thumb_tr_place(web_post_bl()),
            ]
        )
    )

    hulls.append(
        triangle_hulls(
            [
                tbcj_place(tbcj_web_post(6)),
                tbcj_thumb_tr_place(web_post_bl()),
                tbcj_place(tbcj_web_post(7)),
                tbcj_thumb_tr_place(web_post_br()),
                tbcj_place(tbcj_web_post(0)),
                tbcj_thumb_tr_place(web_post_br()),
                key_place(web_post_bl(), 3, lastrow),
            ]
        )
    )

    return geometry_engine.union(hulls)
    # return add(hulls)


##########
## Case ##
##########

def left_key_position(row, direction, low_corner=False, side='right'):
    logging.debug("left_key_position()")
    pos = np.array(
        key_position([-mount_width * 0.5, direction * mount_height * 0.5, 0], 0, row)
    )
    if trackball_in_wall and (side == ball_side or ball_side == 'both'):

        if low_corner:
            x_offset = tbiw_left_wall_lower_x_offset
            y_offset = tbiw_left_wall_lower_y_offset
            z_offset = tbiw_left_wall_lower_z_offset
        else:
            x_offset = 0.0
            y_offset = 0.0
            z_offset = 0.0

        return list(pos - np.array([
            tbiw_left_wall_x_offset_override - x_offset,
            -y_offset,
            tbiw_left_wall_z_offset_override + z_offset
        ]))

    if low_corner:
        x_offset = left_wall_lower_x_offset
        y_offset = left_wall_lower_y_offset
        z_offset = left_wall_lower_z_offset
    else:
        x_offset = 0.0
        y_offset = 0.0
        z_offset = 0.0

    return list(pos - np.array([left_wall_x_offset - x_offset, -y_offset, left_wall_z_offset + z_offset]))


def left_key_place(shape, row, direction, low_corner=False, side='right'):
    logging.debug("left_key_place()")
    pos = left_key_position(row, direction, low_corner=low_corner, side=side)
    return geometry_engine.translate(shape, pos)


def wall_locate1(dx, dy):
    logging.debug("wall_locate1()")
    return [dx * wall_thickness, dy * wall_thickness, -1]


def wall_locate2(dx, dy):
    logging.debug("wall_locate2()")
    return [dx * wall_x_offset, dy * wall_y_offset, -wall_z_offset]


def wall_locate3(dx, dy, back=False):
    logging.debug("wall_locate3()")
    if back:
        return [
            dx * (wall_x_offset + wall_base_x_thickness),
            dy * (wall_y_offset + wall_base_back_thickness),
            -wall_z_offset,
        ]
    else:
        return [
            dx * (wall_x_offset + wall_base_x_thickness),
            dy * (wall_y_offset + wall_base_y_thickness),
            -wall_z_offset,
        ]


def wall_brace(place1, dx1, dy1, post1, place2, dx2, dy2, post2, back=False, skeleton=False, skel_bottom=False):
    logging.debug("wall_brace()")
    hulls = []

    hulls.append(place1(post1))
    if not skeleton:
        hulls.append(place1(geometry_engine.translate(post1, wall_locate1(dx1, dy1))))
        hulls.append(place1(geometry_engine.translate(post1, wall_locate2(dx1, dy1))))
    if not skeleton or skel_bottom:
        hulls.append(place1(geometry_engine.translate(post1, wall_locate3(dx1, dy1, back))))

    hulls.append(place2(post2))
    if not skeleton:
        hulls.append(place2(geometry_engine.translate(post2, wall_locate1(dx2, dy2))))
        hulls.append(place2(geometry_engine.translate(post2, wall_locate2(dx2, dy2))))

    if not skeleton or skel_bottom:
        hulls.append(place2(geometry_engine.translate(post2, wall_locate3(dx2, dy2, back))))

    shape1 = geometry_engine.convex_hull(hulls)

    hulls = []
    if not skeleton:
        hulls.append(place1(geometry_engine.translate(post1, wall_locate2(dx1, dy1))))
    if not skeleton or skel_bottom:
        hulls.append(place1(geometry_engine.translate(post1, wall_locate3(dx1, dy1, back))))
    if not skeleton:
        hulls.append(place2(geometry_engine.translate(post2, wall_locate2(dx2, dy2))))
    if not skeleton or skel_bottom:
        hulls.append(place2(geometry_engine.translate(post2, wall_locate3(dx2, dy2, back))))

    if len(hulls) > 0:
        shape2 = bottom_hull(hulls)
        shape1 = geometry_engine.union([shape1, shape2])
        #shape1 = add([shape1, shape2])

    return shape1


def key_wall_brace(x1, y1, dx1, dy1, post1, x2, y2, dx2, dy2, post2, back=False, skeleton=False, skel_bottom=False):
    logging.debug("key_wall_brace()")
    return wall_brace(
        (lambda shape: key_place(shape, x1, y1)),
        dx1,
        dy1,
        post1,
        (lambda shape: key_place(shape, x2, y2)),
        dx2,
        dy2,
        post2,
        back,
        skeleton=skeleton,
        skel_bottom=False,
    )


def back_wall(skeleton=False):
    logging.debug("back_wall()")
    x = 0
    shape = None
    shape = geometry_engine.union([shape, key_wall_brace(
        x, 0, 0, 1, web_post_tl(), x, 0, 0, 1, web_post_tr(), back=True,
    )])
    for i in range(ncols - 1):
        x = i + 1
        shape = geometry_engine.union([shape, key_wall_brace(
            x, 0, 0, 1, web_post_tl(), x, 0, 0, 1, web_post_tr(), back=True,
        )])

        skelly = skeleton and not x == 1
        shape = geometry_engine.union([shape, key_wall_brace(
            x, 0, 0, 1, web_post_tl(), x - 1, 0, 0, 1, web_post_tr(), back=True,
            skeleton=skelly, skel_bottom=True,
        )])

    shape = geometry_engine.union([shape, key_wall_brace(
        lastcol, 0, 0, 1, web_post_tr(), lastcol, 0, 1, 0, web_post_tr(), back=True,
        skeleton=skeleton, skel_bottom=True,
    )])
    if not skeleton:
        shape = geometry_engine.union([shape,
                                       key_wall_brace(
                                           lastcol, 0, 0, 1, web_post_tr(), lastcol, 0, 1, 0, web_post_tr()
                                       )
                                       ])
    return shape


def right_wall(skeleton=False):
    logging.debug("right_wall()")
    y = 0

    shape = None

    corner = cornerrow if reduced_outer_cols > 0 else lastrow

    shape = geometry_engine.union([shape, key_wall_brace(
        lastcol, y, 1, 0, web_post_tr(), lastcol, y, 1, 0, web_post_br(),
        skeleton=skeleton,
    )])

    for i in range(corner):
        y = i + 1
        shape = geometry_engine.union([shape, key_wall_brace(
            lastcol, y - 1, 1, 0, web_post_br(), lastcol, y, 1, 0, web_post_tr(),
            skeleton=skeleton,
        )])

        shape = geometry_engine.union([shape, key_wall_brace(
            lastcol, y, 1, 0, web_post_tr(), lastcol, y, 1, 0, web_post_br(),
            skeleton=skeleton,
        )])
        # STRANGE PARTIAL OFFSET

    shape = geometry_engine.union([
        shape,
        key_wall_brace(
            lastcol, corner, 0, -1, web_post_br(), lastcol, corner, 1, 0, web_post_br(),
            skeleton=skeleton
        ),
    ])

    return shape


def left_wall(side='right', skeleton=False):
    logging.debug("left_wall()")
    shape = geometry_engine.union([wall_brace(
        (lambda sh: key_place(sh, 0, 0)), 0, 1, web_post_tl(),
        (lambda sh: left_key_place(sh, 0, 1, side=side)), 0, 1, web_post(),
    )])

    shape = geometry_engine.union([shape, wall_brace(
        (lambda sh: left_key_place(sh, 0, 1, side=side)), 0, 1, web_post(),
        (lambda sh: left_key_place(sh, 0, 1, side=side)), -1, 0, web_post(),
        skeleton=skeleton,
    )])

    corner = cornerrow if reduced_inner_cols > 0 else lastrow

    for i in range(corner+1):
        y = i
        low = (y == (corner))
        temp_shape1 = wall_brace(
            (lambda sh: left_key_place(sh, y, 1, side=side)), -1, 0, web_post(),
            (lambda sh: left_key_place(sh, y, -1, low_corner=low, side=side)), -1, 0, web_post(),
            skeleton=skeleton and (y < (corner)),
        )
        shape = geometry_engine.union([shape, temp_shape1])

        temp_shape2 = geometry_engine.convex_hull((
            key_place(web_post_tl(), 0, y),
            key_place(web_post_bl(), 0, y),
            left_key_place(web_post(), y, 1, side=side),
            left_key_place(web_post(), y, -1, low_corner=low, side=side),
        ))

        shape = geometry_engine.union([shape, temp_shape2])

    for i in range(corner):
        y = i + 1
        low = (y == (corner))
        temp_shape1 = wall_brace(
            (lambda sh: left_key_place(sh, y - 1, -1, side=side)), -1, 0, web_post(),
            (lambda sh: left_key_place(sh, y, 1, side=side)), -1, 0, web_post(),
            skeleton=skeleton and (y < (corner)),
        )
        shape = geometry_engine.union([shape, temp_shape1])

        temp_shape2 = geometry_engine.convex_hull((
            key_place(web_post_tl(), 0, y),
            key_place(web_post_bl(), 0, y - 1),
            left_key_place(web_post(), y, 1, side=side),
            left_key_place(web_post(), y - 1, -1, side=side),
        ))

        shape = geometry_engine.union([shape, temp_shape2])

    return shape


def front_wall(skeleton=False):
    logging.debug("front_wall()")
    shape = None

    # shape = geometry_engine.union([shape,key_wall_brace(
    #     3, lastrow, 0, -1, web_post_bl(), 3, lastrow, 0.5, -1, web_post_br()
    # )])
    # shape = geometry_engine.union([shape,key_wall_brace(
    #     3, lastrow, 0.5, -1, web_post_br(), 4, cornerrow, .5, -1, web_post_bl()
    # )])
    # shape = geometry_engine.union([shape,key_wall_brace(
    #     4, cornerrow, .5, -1, web_post_bl(), 4, cornerrow, 0, -1, web_post_br()
    # )])

    # for i in range(ncols - 5):
    #     x = i + 5
    #
    #     shape = geometry_engine.union([shape,key_wall_brace(
    #         x, cornerrow, 0, -1, web_post_bl(), x, cornerrow, 0, -1, web_post_br()
    #     )])
    #
    #     shape = geometry_engine.union([shape, key_wall_brace(
    #         x, cornerrow, 0, -1, web_post_bl(), x - 1, cornerrow, 0, -1, web_post_br()
    #     )])

    # corner = lastrow if 4 < (ncols - reduced_outer_cols) else cornerrow
    corner = cornerrow
    if reduced_outer_cols > 0:
        offset_col = ncols - reduced_outer_cols
    else:
        offset_col = 99

    for i in range(ncols - 3):
        x = i + 3
        logging.debug("col {}", x)
        if x < (offset_col - 1):
            logging.debug("pre-offset")
            if x > 3:
                shape = geometry_engine.union([shape, key_wall_brace(
                    x-1, lastrow, 0, -1, web_post_br(), x, lastrow, 0, -1, web_post_bl()
                )])
            shape = geometry_engine.union([shape, key_wall_brace(
                x, lastrow, 0, -1, web_post_bl(), x, lastrow, 0, -1, web_post_br()
            )])
        elif x < (offset_col):
            logging.debug("offset setup")
            if x > 3:
                shape = geometry_engine.union([shape, key_wall_brace(
                    x-1, lastrow, 0, -1, web_post_br(), x, lastrow, 0, -1, web_post_bl()
                )])
            shape = geometry_engine.union([shape, key_wall_brace(
                x, lastrow, 0, -1, web_post_bl(), x, lastrow, 0.5, -1, web_post_br()
            )])

        elif x == (offset_col):
            logging.debug("offset")
            shape = geometry_engine.union([shape, key_wall_brace(
                x - 1, lastrow, 0.5, -1, web_post_br(), x, cornerrow, .5, -1, web_post_bl()
            )])
            shape = geometry_engine.union([shape, key_wall_brace(
                x, cornerrow, .5, -1, web_post_bl(), x, cornerrow, 0, -1, web_post_br()
            )])

        elif x == (offset_col + 1):
            logging.debug("offset completion")
            shape = geometry_engine.union([shape, key_wall_brace(
                x, cornerrow, 0, -1, web_post_bl(), x - 1, cornerrow, 0, -1, web_post_br()
            )])
            shape = geometry_engine.union([shape, key_wall_brace(
                x, cornerrow, 0, -1, web_post_bl(), x, cornerrow, 0, -1, web_post_br()
            )])

        else:
            logging.debug("post offset")
            shape = geometry_engine.union([shape, key_wall_brace(
                x, cornerrow, 0, -1, web_post_bl(), x - 1, corner, 0, -1, web_post_br()
            )])
            shape = geometry_engine.union([shape, key_wall_brace(
                x, cornerrow, 0, -1, web_post_bl(), x, corner, 0, -1, web_post_br()
            )])

    return shape


def thumb_walls(side='right', style_override=None, skeleton=False):
    if style_override is None:
        _thumb_style = thumb_style
    else:
        _thumb_style = style_override

    if _thumb_style == "MINI":
        return mini_thumb_walls(skeleton=skeleton)
    elif _thumb_style == "MINIDOX":
        return minidox_thumb_walls(skeleton=skeleton)
    elif _thumb_style == "CARBONFET":
        return carbonfet_thumb_walls(skeleton=skeleton)

    elif "TRACKBALL" in _thumb_style:
        if (side == ball_side or ball_side == 'both'):
            if _thumb_style == "TRACKBALL_ORBYL":
                return tbjs_thumb_walls(skeleton=skeleton)
            elif thumb_style == "TRACKBALL_CJ":
                return tbcj_thumb_walls(skeleton=skeleton)

        else:
            return thumb_walls(side, style_override=other_thumb, skeleton=skeleton)
    else:
        return default_thumb_walls(skeleton=skeleton)


def thumb_connection(side='right', style_override=None, skeleton=False):
    if style_override is None:
        _thumb_style = thumb_style
    else:
        _thumb_style = style_override

    if _thumb_style == "MINI":
        return mini_thumb_connection(side=side, skeleton=skeleton)
    elif _thumb_style == "MINIDOX":
        return minidox_thumb_connection(side=side, skeleton=skeleton)
    elif _thumb_style == "CARBONFET":
        return carbonfet_thumb_connection(side=side, skeleton=skeleton)

    elif "TRACKBALL" in _thumb_style:
        if (side == ball_side or ball_side == 'both'):
            if _thumb_style == "TRACKBALL_ORBYL":
                return tbjs_thumb_connection(side=side, skeleton=skeleton)
            elif thumb_style == "TRACKBALL_CJ":
                return tbcj_thumb_connection(side=side, skeleton=skeleton)
        else:
            return thumb_connection(side, style_override=other_thumb, skeleton=skeleton)
    else:
        return default_thumb_connection(side=side, skeleton=skeleton)


def default_thumb_walls(skeleton=False):
    logging.debug("thumb_walls()")
    # thumb, walls
    if default_1U_cluster:
        shape = geometry_engine.union([wall_brace(default_thumb_mr_place, 0, -1, web_post_br(), default_thumb_tr_place, 0, -1, web_post_br())])
    else:
        shape = geometry_engine.union([wall_brace(default_thumb_mr_place, 0, -1, web_post_br(), default_thumb_tr_place, 0, -1, thumb_post_br())])
    shape = geometry_engine.union([shape, wall_brace(default_thumb_mr_place, 0, -1, web_post_br(), default_thumb_mr_place, 0, -1, web_post_bl())])
    shape = geometry_engine.union([shape, wall_brace(default_thumb_br_place, 0, -1, web_post_br(), default_thumb_br_place, 0, -1, web_post_bl())])
    shape = geometry_engine.union([shape, wall_brace(default_thumb_ml_place, -0.3, 1, web_post_tr(), default_thumb_ml_place, 0, 1, web_post_tl())])
    shape = geometry_engine.union([shape, wall_brace(default_thumb_bl_place, 0, 1, web_post_tr(), default_thumb_bl_place, 0, 1, web_post_tl())])
    shape = geometry_engine.union([shape, wall_brace(default_thumb_br_place, -1, 0, web_post_tl(), default_thumb_br_place, -1, 0, web_post_bl())])
    shape = geometry_engine.union([shape, wall_brace(default_thumb_bl_place, -1, 0, web_post_tl(), default_thumb_bl_place, -1, 0, web_post_bl())])
    # thumb, corners
    shape = geometry_engine.union([shape, wall_brace(default_thumb_br_place, -1, 0, web_post_bl(), default_thumb_br_place, 0, -1, web_post_bl())])
    shape = geometry_engine.union([shape, wall_brace(default_thumb_bl_place, -1, 0, web_post_tl(), default_thumb_bl_place, 0, 1, web_post_tl())])
    # thumb, tweeners
    shape = geometry_engine.union([shape, wall_brace(default_thumb_mr_place, 0, -1, web_post_bl(), default_thumb_br_place, 0, -1, web_post_br())])
    shape = geometry_engine.union([shape, wall_brace(default_thumb_ml_place, 0, 1, web_post_tl(), default_thumb_bl_place, 0, 1, web_post_tr())])
    shape = geometry_engine.union([shape, wall_brace(default_thumb_bl_place, -1, 0, web_post_bl(), default_thumb_br_place, -1, 0, web_post_tl())])
    if default_1U_cluster:
        shape = geometry_engine.union([shape, wall_brace(default_thumb_tr_place, 0, -1, web_post_br(),
                                      (lambda sh: key_place(sh, 3, lastrow)), 0, -1, web_post_bl())])
    else:
        shape = geometry_engine.union([shape, wall_brace(default_thumb_tr_place, 0, -1, thumb_post_br(),
                                      (lambda sh: key_place(sh, 3, lastrow)), 0, -1, web_post_bl())])

    return shape


def default_thumb_connection(side='right', skeleton=False):
    logging.debug("thumb_connection()")
    # clunky bit on the top left thumb connection  (normal connectors don't work well)
    shape = None
    shape = geometry_engine.union([shape, bottom_hull(
        [
            left_key_place(geometry_engine.translate(web_post(), wall_locate2(-1, 0)), cornerrow, -1, low_corner=True, side=side),
            left_key_place(geometry_engine.translate(web_post(), wall_locate3(-1, 0)), cornerrow, -1, low_corner=True, side=side),
            default_thumb_ml_place(geometry_engine.translate(web_post_tr(), wall_locate2(-0.3, 1))),
            default_thumb_ml_place(geometry_engine.translate(web_post_tr(), wall_locate3(-0.3, 1))),
        ]
    )])

    shape = geometry_engine.union([shape,
                                   geometry_engine.convex_hull(
                                       [
                                           left_key_place(geometry_engine.translate(web_post(), wall_locate2(-1, 0)),
                                                          cornerrow, -1, low_corner=True, side=side),
                                           left_key_place(geometry_engine.translate(web_post(), wall_locate3(-1, 0)),
                                                          cornerrow, -1, low_corner=True, side=side),
                                           default_thumb_ml_place(geometry_engine.translate(web_post_tr(), wall_locate2(-0.3, 1))),
                                           default_thumb_ml_place(geometry_engine.translate(web_post_tr(), wall_locate3(-0.3, 1))),
                                           default_thumb_tl_place(thumb_post_tl()),
                                       ]
                                   )
                                   ])  # )

    shape = geometry_engine.union([shape, geometry_engine.convex_hull(
        [
            left_key_place(geometry_engine.translate(web_post(), wall_locate1(-1, 0)), cornerrow, -1, low_corner=True, side=side),
            left_key_place(geometry_engine.translate(web_post(), wall_locate2(-1, 0)), cornerrow, -1, low_corner=True, side=side),
            left_key_place(geometry_engine.translate(web_post(), wall_locate3(-1, 0)), cornerrow, -1, low_corner=True, side=side),
            default_thumb_tl_place(thumb_post_tl()),
        ]
    )])

    shape = geometry_engine.union([shape, geometry_engine.convex_hull(
        [
            left_key_place(web_post(), cornerrow, -1, low_corner=True, side=side),
            left_key_place(geometry_engine.translate(web_post(), wall_locate1(-1, 0)), cornerrow, -1, low_corner=True, side=side),
            key_place(web_post_bl(), 0, cornerrow),
            default_thumb_tl_place(thumb_post_tl()),
        ]
    )])

    shape = geometry_engine.union([shape, geometry_engine.convex_hull(
        [
            default_thumb_ml_place(web_post_tr()),
            default_thumb_ml_place(geometry_engine.translate(web_post_tr(), wall_locate1(-0.3, 1))),
            default_thumb_ml_place(geometry_engine.translate(web_post_tr(), wall_locate2(-0.3, 1))),
            default_thumb_ml_place(geometry_engine.translate(web_post_tr(), wall_locate3(-0.3, 1))),
            default_thumb_tl_place(thumb_post_tl()),
        ]
    )])

    return shape


def tbjs_thumb_connection(side='right', skeleton=False):
    logging.debug("thumb_connection()")
    # clunky bit on the top left thumb connection  (normal connectors don't work well)
    hulls = []
    hulls.append(
        triangle_hulls(
            [
                key_place(web_post_bl(), 0, cornerrow),
                left_key_place(web_post(), cornerrow, -1, side=side, low_corner=True),
                # left_key_place(geometry_engine.translate(web_post(), wall_locate1(-1, 0)), cornerrow, -1, low_corner=True),
                tbjs_place(tbjs_post_tl()),
            ]
        )
    )

    hulls.append(
        triangle_hulls(
            [
                key_place(web_post_bl(), 0, cornerrow),
                tbjs_thumb_tl_place(tbjs_thumb_post_bl()),
                key_place(web_post_br(), 0, cornerrow),
                tbjs_thumb_tl_place(tbjs_thumb_post_tl()),
                key_place(web_post_bl(), 1, cornerrow),
                tbjs_thumb_tl_place(tbjs_thumb_post_tl()),
                key_place(web_post_br(), 1, cornerrow),
                tbjs_thumb_tl_place(tbjs_thumb_post_tr()),
                key_place(web_post_bl(), 2, lastrow),
                tbjs_thumb_tl_place(tbjs_thumb_post_tr()),
                key_place(web_post_bl(), 2, lastrow),
                tbjs_thumb_mr_place(tbjs_thumb_post_tl()),
                key_place(web_post_br(), 2, lastrow),
                key_place(web_post_bl(), 3, lastrow),
                tbjs_thumb_mr_place(tbjs_thumb_post_tr()),
                tbjs_thumb_mr_place(tbjs_thumb_post_tl()),
                key_place(web_post_br(), 2, lastrow),

            ]
        )
    )
    shape = geometry_engine.union(hulls)
    return shape


def tbjs_thumb_walls(skeleton=False):
    logging.debug("thumb_walls()")
    # thumb, walls
    shape = wall_brace(
        tbjs_thumb_mr_place, .5, 1, tbjs_thumb_post_tr(),
        (lambda sh: key_place(sh, 3, lastrow)), 0, -1, web_post_bl(),
    )
    shape = geometry_engine.union([shape, wall_brace(
        tbjs_thumb_mr_place, .5, 1, tbjs_thumb_post_tr(),
        tbjs_thumb_br_place, 0, -1, tbjs_thumb_post_br(),
    )])
    shape = geometry_engine.union([shape, wall_brace(
        tbjs_thumb_br_place, 0, -1, tbjs_thumb_post_br(),
        tbjs_thumb_br_place, 0, -1, tbjs_thumb_post_bl(),
    )])
    shape = geometry_engine.union([shape, wall_brace(
        tbjs_thumb_br_place, 0, -1, tbjs_thumb_post_bl(),
        tbjs_thumb_bl_place, 0, -1, tbjs_thumb_post_br(),
    )])
    shape = geometry_engine.union([shape, wall_brace(
        tbjs_thumb_bl_place, 0, -1, tbjs_thumb_post_br(),
        tbjs_thumb_bl_place, -1, -1, tbjs_thumb_post_bl(),
    )])

    shape = geometry_engine.union([shape, wall_brace(
        tbjs_place, -1.5, 0, tbjs_post_tl(),
        (lambda sh: left_key_place(sh, cornerrow, -1, side=ball_side, low_corner=True)), -1, 0, web_post(),
    )])
    shape = geometry_engine.union([shape, wall_brace(
        tbjs_place, -1.5, 0, tbjs_post_tl(),
        tbjs_place, -1, 0, tbjs_post_l(),
    )])
    shape = geometry_engine.union([shape, wall_brace(
        tbjs_place, -1, 0, tbjs_post_l(),
        tbjs_thumb_bl_place, -1, 0, tbjs_thumb_post_tl(),
    )])
    shape = geometry_engine.union([shape, wall_brace(
        tbjs_thumb_bl_place, -1, 0, tbjs_thumb_post_tl(),
        tbjs_thumb_bl_place, -1, -1, tbjs_thumb_post_bl(),
    )])

    return shape


def tbcj_thumb_connection(side='right', skeleton=False):
    # clunky bit on the top left thumb connection  (normal connectors don't work well)
    shape = geometry_engine.union([bottom_hull(
        [
            left_key_place(geometry_engine.translate(web_post(), wall_locate2(-1, 0)), cornerrow, -1, low_corner=True, side=side),
            left_key_place(geometry_engine.translate(web_post(), wall_locate3(-1, 0)), cornerrow, -1, low_corner=True, side=side),
            default_thumb_ml_place(geometry_engine.translate(web_post_tr(), wall_locate2(-0.3, 1))),
            default_thumb_ml_place(geometry_engine.translate(web_post_tr(), wall_locate3(-0.3, 1))),
        ]
    )])

    shape = geometry_engine.union([shape,
                                   geometry_engine.convex_hull(
                                       [
                                           left_key_place(geometry_engine.translate(web_post(), wall_locate2(-1, 0)),
                                                          cornerrow, -1, low_corner=True, side=side),
                                           left_key_place(geometry_engine.translate(web_post(), wall_locate3(-1, 0)),
                                                          cornerrow, -1, low_corner=True, side=side),
                                           default_thumb_ml_place(geometry_engine.translate(web_post_tr(), wall_locate2(-0.3, 1))),
                                           default_thumb_ml_place(geometry_engine.translate(web_post_tr(), wall_locate3(-0.3, 1))),
                                           default_thumb_tl_place(web_post_tl()),
                                       ]
                                   )
                                   ])  # )

    shape = geometry_engine.union([shape, geometry_engine.convex_hull(
        [
            left_key_place(geometry_engine.translate(web_post(), wall_locate1(-1, 0)), cornerrow, -1, low_corner=True, side=side),
            left_key_place(geometry_engine.translate(web_post(), wall_locate2(-1, 0)), cornerrow, -1, low_corner=True, side=side),
            left_key_place(geometry_engine.translate(web_post(), wall_locate3(-1, 0)), cornerrow, -1, low_corner=True, side=side),
            default_thumb_tl_place(web_post_tl()),
        ]
    )])

    shape = geometry_engine.union([shape, geometry_engine.convex_hull(
        [
            left_key_place(web_post(), cornerrow, -1, low_corner=True, side=side),
            left_key_place(geometry_engine.translate(web_post(), wall_locate1(-1, 0)), cornerrow, -1, low_corner=True, side=side),
            key_place(web_post_bl(), 0, cornerrow),
            default_thumb_tl_place(web_post_tl()),
        ]
    )])

    shape = geometry_engine.union([shape, geometry_engine.convex_hull(
        [
            default_thumb_ml_place(web_post_tr()),
            default_thumb_ml_place(geometry_engine.translate(web_post_tr(), wall_locate1(-0.3, 1))),
            default_thumb_ml_place(geometry_engine.translate(web_post_tr(), wall_locate2(-0.3, 1))),
            default_thumb_ml_place(geometry_engine.translate(web_post_tr(), wall_locate3(-0.3, 1))),
            default_thumb_tl_place(web_post_tl()),
        ]
    )])

    return shape


def tbcj_thumb_walls(skeleton=False):
    shape = geometry_engine.union([wall_brace(tbcj_thumb_ml_place, -0.3, 1, web_post_tr(), tbcj_thumb_ml_place, 0, 1, web_post_tl())])
    shape = geometry_engine.union([shape, wall_brace(tbcj_thumb_bl_place, 0, 1, web_post_tr(), tbcj_thumb_bl_place, 0, 1, web_post_tl())])
    shape = geometry_engine.union([shape, wall_brace(tbcj_thumb_bl_place, -1, 0, web_post_tl(), tbcj_thumb_bl_place, -1, 0, web_post_bl())])
    shape = geometry_engine.union([shape, wall_brace(tbcj_thumb_bl_place, -1, 0, web_post_tl(), tbcj_thumb_bl_place, 0, 1, web_post_tl())])
    shape = geometry_engine.union([shape, wall_brace(tbcj_thumb_ml_place, 0, 1, web_post_tl(), tbcj_thumb_bl_place, 0, 1, web_post_tr())])

    corner = geometry_engine.box(1, 1, tbcj_thickness)

    points = [
        (tbcj_thumb_bl_place, -1, 0, web_post_bl()),
        (tbcj_place, 0, -1, tbcj_web_post(4)),
        (tbcj_place, 0, -1, tbcj_web_post(3)),
        (tbcj_place, 0, -1, tbcj_web_post(2)),
        (tbcj_place, 1, -1, tbcj_web_post(1)),
        (tbcj_place, 1, 0, tbcj_web_post(0)),
        ((lambda sh: key_place(sh, 3, lastrow)), 0, -1, web_post_bl()),
    ]
    for i, _ in enumerate(points[:-1]):
        (pa, dxa, dya, sa) = points[i]
        (pb, dxb, dyb, sb) = points[i + 1]

        shape = geometry_engine.union([shape, wall_brace(pa, dxa, dya, sa, pb, dxb, dyb, sb)])

    return shape


def mini_thumb_walls(skeleton=False):
    # thumb, walls
    shape = geometry_engine.union([wall_brace(mini_thumb_mr_place, 0, -1, web_post_br(), mini_thumb_tr_place, 0, -1, mini_thumb_post_br())])
    shape = geometry_engine.union([shape, wall_brace(mini_thumb_mr_place, 0, -1, web_post_br(), mini_thumb_mr_place, 0, -1, web_post_bl())])
    shape = geometry_engine.union([shape, wall_brace(mini_thumb_br_place, 0, -1, web_post_br(), mini_thumb_br_place, 0, -1, web_post_bl())])
    shape = geometry_engine.union([shape, wall_brace(mini_thumb_bl_place, 0, 1, web_post_tr(), mini_thumb_bl_place, 0, 1, web_post_tl())])
    shape = geometry_engine.union([shape, wall_brace(mini_thumb_br_place, -1, 0, web_post_tl(), mini_thumb_br_place, -1, 0, web_post_bl())])
    shape = geometry_engine.union([shape, wall_brace(mini_thumb_bl_place, -1, 0, web_post_tl(), mini_thumb_bl_place, -1, 0, web_post_bl())])
    # thumb, corners
    shape = geometry_engine.union([shape, wall_brace(mini_thumb_br_place, -1, 0, web_post_bl(), mini_thumb_br_place, 0, -1, web_post_bl())])
    shape = geometry_engine.union([shape, wall_brace(mini_thumb_bl_place, -1, 0, web_post_tl(), mini_thumb_bl_place, 0, 1, web_post_tl())])
    # thumb, tweeners
    shape = geometry_engine.union([shape, wall_brace(mini_thumb_mr_place, 0, -1, web_post_bl(), mini_thumb_br_place, 0, -1, web_post_br())])
    shape = geometry_engine.union([shape, wall_brace(mini_thumb_bl_place, -1, 0, web_post_bl(), mini_thumb_br_place, -1, 0, web_post_tl())])
    shape = geometry_engine.union([shape, wall_brace(mini_thumb_tr_place, 0, -1, mini_thumb_post_br(),
                                  (lambda sh: key_place(sh, 3, lastrow)), 0, -1, web_post_bl())])

    return shape


def mini_thumb_connection(side='right', skeleton=False):
    # clunky bit on the top left thumb connection  (normal connectors don't work well)
    shape = geometry_engine.union([bottom_hull(
        [
            left_key_place(geometry_engine.translate(web_post(), wall_locate2(-1, 0)), cornerrow, -1, low_corner=True, side=side),
            left_key_place(geometry_engine.translate(web_post(), wall_locate3(-1, 0)), cornerrow, -1, low_corner=True, side=side),
            mini_thumb_bl_place(geometry_engine.translate(web_post_tr(), wall_locate2(-0.3, 1))),
            mini_thumb_bl_place(geometry_engine.translate(web_post_tr(), wall_locate3(-0.3, 1))),
        ]
    )])

    shape = geometry_engine.union([shape,
                                   geometry_engine.convex_hull(
                                       [
                                           left_key_place(geometry_engine.translate(web_post(), wall_locate2(-1, 0)),
                                                          cornerrow, -1, low_corner=True, side=side),
                                           left_key_place(geometry_engine.translate(web_post(), wall_locate3(-1, 0)),
                                                          cornerrow, -1, low_corner=True, side=side),
                                           mini_thumb_bl_place(geometry_engine.translate(web_post_tr(), wall_locate2(-0.3, 1))),
                                           mini_thumb_bl_place(geometry_engine.translate(web_post_tr(), wall_locate3(-0.3, 1))),
                                           mini_thumb_tl_place(web_post_tl()),
                                       ]
                                   )])

    shape = geometry_engine.union([shape,
                                   geometry_engine.convex_hull(
                                       [
                                           left_key_place(web_post(), cornerrow, -1, low_corner=True, side=side),
                                           left_key_place(geometry_engine.translate(web_post(), wall_locate1(-1, 0)),
                                                          cornerrow, -1, low_corner=True, side=side),
                                           left_key_place(geometry_engine.translate(web_post(), wall_locate2(-1, 0)),
                                                          cornerrow, -1, low_corner=True, side=side),
                                           left_key_place(geometry_engine.translate(web_post(), wall_locate3(-1, 0)),
                                                          cornerrow, -1, low_corner=True, side=side),
                                           mini_thumb_tl_place(web_post_tl()),
                                       ]
                                   )])

    shape = geometry_engine.union([shape,
                                   geometry_engine.convex_hull(
                                       [
                                           left_key_place(web_post(), cornerrow, -1, low_corner=True, side=side),
                                           left_key_place(geometry_engine.translate(web_post(), wall_locate1(-1, 0)),
                                                          cornerrow, -1, low_corner=True, side=side),
                                           key_place(web_post_bl(), 0, cornerrow),
                                           mini_thumb_tl_place(web_post_tl()),
                                       ]
                                   )])

    shape = geometry_engine.union([shape,
                                   geometry_engine.convex_hull(
                                       [
                                           mini_thumb_bl_place(web_post_tr()),
                                           mini_thumb_bl_place(geometry_engine.translate(web_post_tr(), wall_locate1(-0.3, 1))),
                                           mini_thumb_bl_place(geometry_engine.translate(web_post_tr(), wall_locate2(-0.3, 1))),
                                           mini_thumb_bl_place(geometry_engine.translate(web_post_tr(), wall_locate3(-0.3, 1))),
                                           mini_thumb_tl_place(web_post_tl()),
                                       ]
                                   )])

    return shape


def minidox_thumb_walls(skeleton=False):

    # thumb, walls
    shape = geometry_engine.union([wall_brace(minidox_thumb_tr_place, 0, -1, minidox_thumb_post_br(), minidox_thumb_tr_place, 0, -1, minidox_thumb_post_bl())])
    shape = geometry_engine.union([shape, wall_brace(minidox_thumb_tr_place, 0, -1, minidox_thumb_post_bl(),
                                  minidox_thumb_tl_place, 0, -1, minidox_thumb_post_br())])
    shape = geometry_engine.union([shape, wall_brace(minidox_thumb_tl_place, 0, -1, minidox_thumb_post_br(),
                                  minidox_thumb_tl_place, 0, -1, minidox_thumb_post_bl())])
    shape = geometry_engine.union([shape, wall_brace(minidox_thumb_tl_place, 0, -1, minidox_thumb_post_bl(),
                                  minidox_thumb_ml_place, -1, -1, minidox_thumb_post_br())])
    shape = geometry_engine.union([shape, wall_brace(minidox_thumb_ml_place, -1, -1, minidox_thumb_post_br(),
                                  minidox_thumb_ml_place, 0, -1, minidox_thumb_post_bl())])
    shape = geometry_engine.union([shape, wall_brace(minidox_thumb_ml_place, 0, -1, minidox_thumb_post_bl(),
                                  minidox_thumb_ml_place, -1, 0, minidox_thumb_post_bl())])
    # thumb, corners
    shape = geometry_engine.union([shape, wall_brace(minidox_thumb_ml_place, -1, 0, minidox_thumb_post_bl(),
                                  minidox_thumb_ml_place, -1, 0, minidox_thumb_post_tl())])
    shape = geometry_engine.union([shape, wall_brace(minidox_thumb_ml_place, -1, 0, minidox_thumb_post_tl(),
                                  minidox_thumb_ml_place, 0, 1, minidox_thumb_post_tl())])
    # thumb, tweeners
    shape = geometry_engine.union([shape, wall_brace(minidox_thumb_ml_place, 0, 1, minidox_thumb_post_tr(),
                                  minidox_thumb_ml_place, 0, 1, minidox_thumb_post_tl())])
    shape = geometry_engine.union([shape, wall_brace(minidox_thumb_tr_place, 0, -1, minidox_thumb_post_br(),
                                  (lambda sh: key_place(sh, 3, lastrow)), 0, -1, web_post_bl())])

    return shape


def minidox_thumb_connection(side='right', skeleton=False):
    # clunky bit on the top left thumb connection  (normal connectors don't work well)
    shape = geometry_engine.union([bottom_hull(
        [
            left_key_place(geometry_engine.translate(web_post(), wall_locate2(-1, 0)), cornerrow, -1, low_corner=True, side=side),
            left_key_place(geometry_engine.translate(web_post(), wall_locate3(-1, 0)), cornerrow, -1, low_corner=True, side=side),
            minidox_thumb_ml_place(geometry_engine.translate(minidox_thumb_post_tr(), wall_locate2(-0.3, 1))),
            minidox_thumb_ml_place(geometry_engine.translate(minidox_thumb_post_tr(), wall_locate3(-0.3, 1))),
        ]
    )])

    shape = geometry_engine.union([shape,
                                   geometry_engine.convex_hull(
                                       [
                                           left_key_place(geometry_engine.translate(web_post(), wall_locate2(-1, 0)),
                                                          cornerrow, -1, low_corner=True, side=side),
                                           left_key_place(geometry_engine.translate(web_post(), wall_locate3(-1, 0)),
                                                          cornerrow, -1, low_corner=True, side=side),
                                           minidox_thumb_ml_place(geometry_engine.translate(minidox_thumb_post_tr(), wall_locate2(-0.3, 1))),
                                           minidox_thumb_ml_place(geometry_engine.translate(minidox_thumb_post_tr(), wall_locate3(-0.3, 1))),
                                           minidox_thumb_tl_place(minidox_thumb_post_tl()),
                                       ]
                                   )])

    shape = geometry_engine.union([shape,
                                   geometry_engine.convex_hull(
                                       [
                                           left_key_place(web_post(), cornerrow, -1, low_corner=True, side=side),
                                           left_key_place(geometry_engine.translate(web_post(), wall_locate1(-1, 0)),
                                                          cornerrow, -1, low_corner=True, side=side),
                                           left_key_place(geometry_engine.translate(web_post(), wall_locate2(-1, 0)),
                                                          cornerrow, -1, low_corner=True, side=side),
                                           left_key_place(geometry_engine.translate(web_post(), wall_locate3(-1, 0)),
                                                          cornerrow, -1, low_corner=True, side=side),
                                           minidox_thumb_tl_place(minidox_thumb_post_tl()),
                                       ]
                                   )])

    shape = geometry_engine.union([shape,
                                   geometry_engine.convex_hull(
                                       [
                                           left_key_place(web_post(), cornerrow, -1, low_corner=True, side=side),
                                           left_key_place(geometry_engine.translate(web_post(), wall_locate1(-1, 0)),
                                                          cornerrow, -1, low_corner=True, side=side),
                                           key_place(web_post_bl(), 0, cornerrow),
                                           minidox_thumb_tl_place(minidox_thumb_post_tl()),
                                       ]
                                   )])

    shape = geometry_engine.union([shape,
                                   geometry_engine.convex_hull(
                                       [
                                           minidox_thumb_ml_place(minidox_thumb_post_tr()),
                                           minidox_thumb_ml_place(geometry_engine.translate(minidox_thumb_post_tr(), wall_locate1(0, 1))),
                                           minidox_thumb_ml_place(geometry_engine.translate(minidox_thumb_post_tr(), wall_locate2(0, 1))),
                                           minidox_thumb_ml_place(geometry_engine.translate(minidox_thumb_post_tr(), wall_locate3(0, 1))),
                                           minidox_thumb_tl_place(minidox_thumb_post_tl()),
                                       ]
                                   )])

    return shape


def carbonfet_thumb_walls(skeleton=False):
    # thumb, walls
    shape = geometry_engine.union([wall_brace(carbonfet_thumb_mr_place, 0, -1, web_post_br(), carbonfet_thumb_tr_place, 0, -1, web_post_br())])
    shape = geometry_engine.union([shape, wall_brace(carbonfet_thumb_mr_place, 0, -1, web_post_br(), carbonfet_thumb_mr_place, 0, -1.15, web_post_bl())])
    shape = geometry_engine.union([shape, wall_brace(carbonfet_thumb_br_place, 0, -1, web_post_br(), carbonfet_thumb_br_place, 0, -1, web_post_bl())])
    shape = geometry_engine.union([shape, wall_brace(carbonfet_thumb_bl_place, -.3, 1, thumb_post_tr(), carbonfet_thumb_bl_place, 0, 1, thumb_post_tl())])
    shape = geometry_engine.union([shape, wall_brace(carbonfet_thumb_br_place, -1, 0, web_post_tl(), carbonfet_thumb_br_place, -1, 0, web_post_bl())])
    shape = geometry_engine.union([shape, wall_brace(carbonfet_thumb_bl_place, -1, 0, thumb_post_tl(), carbonfet_thumb_bl_place, -1, 0, web_post_bl())])
    # thumb, corners
    shape = geometry_engine.union([shape, wall_brace(carbonfet_thumb_br_place, -1, 0, web_post_bl(), carbonfet_thumb_br_place, 0, -1, web_post_bl())])
    shape = geometry_engine.union([shape, wall_brace(carbonfet_thumb_bl_place, -1, 0, thumb_post_tl(), carbonfet_thumb_bl_place, 0, 1, thumb_post_tl())])
    # thumb, tweeners
    shape = geometry_engine.union([shape, wall_brace(carbonfet_thumb_mr_place, 0, -1.15, web_post_bl(), carbonfet_thumb_br_place, 0, -1, web_post_br())])
    shape = geometry_engine.union([shape, wall_brace(carbonfet_thumb_bl_place, -1, 0, web_post_bl(), carbonfet_thumb_br_place, -1, 0, web_post_tl())])
    shape = geometry_engine.union([shape, wall_brace(carbonfet_thumb_tr_place, 0, -1, web_post_br(),
                                  (lambda sh: key_place(sh, 3, lastrow)), 0, -1, web_post_bl())])
    return shape


def carbonfet_thumb_connection(side='right', skeleton=False):
    # clunky bit on the top left thumb connection  (normal connectors don't work well)
    shape = bottom_hull(
        [
            left_key_place(geometry_engine.translate(web_post(), wall_locate2(-1, 0)), cornerrow, -1, low_corner=True, side=side),
            left_key_place(geometry_engine.translate(web_post(), wall_locate3(-1, 0)), cornerrow, -1, low_corner=True, side=side),
            carbonfet_thumb_bl_place(geometry_engine.translate(thumb_post_tr(), wall_locate2(-0.3, 1))),
            carbonfet_thumb_bl_place(geometry_engine.translate(thumb_post_tr(), wall_locate3(-0.3, 1))),
        ]
    )

    shape = geometry_engine.union([shape,
                                   geometry_engine.convex_hull(
                                       [
                                           left_key_place(geometry_engine.translate(web_post(), wall_locate2(-1, 0)),
                                                          cornerrow, -1, low_corner=True, side=side),
                                           left_key_place(geometry_engine.translate(web_post(), wall_locate3(-1, 0)),
                                                          cornerrow, -1, low_corner=True, side=side),
                                           carbonfet_thumb_bl_place(geometry_engine.translate(thumb_post_tr(), wall_locate2(-0.3, 1))),
                                           carbonfet_thumb_bl_place(geometry_engine.translate(thumb_post_tr(), wall_locate3(-0.3, 1))),
                                           carbonfet_thumb_ml_place(thumb_post_tl()),
                                       ]
                                   )])

    shape = geometry_engine.union([shape,
                                   geometry_engine.convex_hull(
                                       [
                                           left_key_place(web_post(), cornerrow, -1, low_corner=True, side=side),
                                           left_key_place(geometry_engine.translate(web_post(), wall_locate1(-1, 0)),
                                                          cornerrow, -1, low_corner=True, side=side),
                                           left_key_place(geometry_engine.translate(web_post(), wall_locate2(-1, 0)),
                                                          cornerrow, -1, low_corner=True, side=side),
                                           left_key_place(geometry_engine.translate(web_post(), wall_locate3(-1, 0)),
                                                          cornerrow, -1, low_corner=True, side=side),
                                           carbonfet_thumb_ml_place(thumb_post_tl()),
                                       ]
                                   )])

    shape = geometry_engine.union([shape,
                                   geometry_engine.convex_hull(
                                       [
                                           left_key_place(web_post(), cornerrow, -1, low_corner=True, side=side),
                                           left_key_place(geometry_engine.translate(web_post(), wall_locate1(-1, 0)),
                                                          cornerrow, -1, low_corner=True, side=side),
                                           key_place(web_post_bl(), 0, cornerrow),
                                           carbonfet_thumb_ml_place(thumb_post_tl()),
                                       ]
                                   )])

    shape = geometry_engine.union([shape,
                                   geometry_engine.convex_hull(
                                       [
                                           carbonfet_thumb_bl_place(thumb_post_tr()),
                                           carbonfet_thumb_bl_place(geometry_engine.translate(thumb_post_tr(), wall_locate1(-0.3, 1))),
                                           carbonfet_thumb_bl_place(geometry_engine.translate(thumb_post_tr(), wall_locate2(-0.3, 1))),
                                           carbonfet_thumb_bl_place(geometry_engine.translate(thumb_post_tr(), wall_locate3(-0.3, 1))),
                                           carbonfet_thumb_ml_place(thumb_post_tl()),
                                       ]
                                   )])

    return shape


def case_walls(side='right', skeleton=False):
    logging.debug("case_walls()")
    return (
        geometry_engine.union([
            back_wall(skeleton=skeleton),
            left_wall(side=side, skeleton=skeleton),
            right_wall(skeleton=skeleton),
            front_wall(skeleton=skeleton),
            # thumb_walls(side=side),
            # thumb_connection(side=side),
        ])
    )


rj9_start = list(
    np.array([0, -3, 0])
    + np.array(
        key_position(
            list(np.array(wall_locate3(0, 1)) + np.array([0, (mount_height / 2), 0])),
            0,
            0,
        )
    )
)

rj9_position = (rj9_start[0], rj9_start[1], 11)


def rj9_cube():
    logging.debug("rj9_cube()")
    shape = geometry_engine.box(14.78, 13, 22.38)

    return shape


def rj9_space():
    logging.debug("rj9_space()")
    return geometry_engine.translate(rj9_cube(), rj9_position)


def rj9_holder():
    logging.debug("rj9_holder()")
    shape = geometry_engine.union([geometry_engine.translate(geometry_engine.box(10.78, 9, 18.38), (0, 2, 0)),
                                  geometry_engine.translate(geometry_engine.box(10.78, 13, 5), (0, 0, 5))])
    shape = geometry_engine.difference(rj9_cube(), [shape])
    shape = geometry_engine.translate(shape, rj9_position)

    return shape


usb_holder_position = key_position(
    list(np.array(wall_locate2(0, 1)) + np.array([0, (mount_height / 2), 0])), 1, 0
)
usb_holder_size = [6.5, 10.0, 13.6]
usb_holder_thickness = 4


def usb_holder():
    logging.debug("usb_holder()")
    shape = geometry_engine.box(
        usb_holder_size[0] + usb_holder_thickness,
        usb_holder_size[1],
        usb_holder_size[2] + usb_holder_thickness,
    )
    shape = geometry_engine.translate(shape,
                                      (
                                          usb_holder_position[0],
                                          usb_holder_position[1],
                                          (usb_holder_size[2] + usb_holder_thickness) / 2,
                                      )
                                      )
    return shape


def usb_holder_hole():
    logging.debug("usb_holder_hole()")
    shape = geometry_engine.box(*usb_holder_size)
    shape = geometry_engine.translate(shape,
                                      (
                                          usb_holder_position[0],
                                          usb_holder_position[1],
                                          (usb_holder_size[2] + usb_holder_thickness) / 2,
                                      )
                                      )
    return shape


external_start = list(
    # np.array([0, -3, 0])
    np.array([external_holder_width / 2, 0, 0])
    + np.array(
        key_position(
            list(np.array(wall_locate3(0, 1)) + np.array([0, (mount_height / 2), 0])),
            0,
            0,
        )
    )
)


def external_mount_hole():
    logging.debug("external_mount_hole()")
    shape = geometry_engine.box(external_holder_width, 20.0, external_holder_height+.1)
    undercut = geometry_engine.box(external_holder_width+8, 10.0, external_holder_height+8+.1)
    shape = geometry_engine.union([shape, geometry_engine.translate(undercut, (0, -5, 0))])

    shape = geometry_engine.translate(shape,
                                      (
                                          external_start[0] + external_holder_xoffset,
                                          external_start[1] + external_holder_yoffset,
                                          external_holder_height / 2-.05,
                                      )
                                      )
    return shape


pcb_mount_ref_position = key_position(
    # TRRS POSITION IS REFERENCE BY CONVENIENCE
    list(np.array(wall_locate3(0, 1)) + np.array([0, (mount_height / 2), 0])), 0, 0
)

pcb_mount_ref_position[0] = pcb_mount_ref_position[0] + pcb_mount_ref_offset[0]
pcb_mount_ref_position[1] = pcb_mount_ref_position[1] + pcb_mount_ref_offset[1]
pcb_mount_ref_position[2] = 0.0 + pcb_mount_ref_offset[2]


def pcb_usb_hole():
    logging.debug("pcb_holder()")
    pcb_usb_position = copy.deepcopy(pcb_mount_ref_position)
    pcb_usb_position[0] = pcb_usb_position[0] + pcb_usb_hole_offset[0]
    pcb_usb_position[1] = pcb_usb_position[1] + pcb_usb_hole_offset[1]
    pcb_usb_position[2] = pcb_usb_position[2] + pcb_usb_hole_offset[2]

    shape = geometry_engine.box(*pcb_usb_hole_size)
    shape = geometry_engine.translate(shape,
                                      (
                                          pcb_usb_position[0],
                                          pcb_usb_position[1],
                                          pcb_usb_hole_size[2] / 2 + usb_holder_thickness,
                                      )
                                      )
    return shape


pcb_holder_position = copy.deepcopy(pcb_mount_ref_position)
pcb_holder_position[0] = pcb_holder_position[0] + pcb_holder_offset[0]
pcb_holder_position[1] = pcb_holder_position[1] + pcb_holder_offset[1]
pcb_holder_position[2] = pcb_holder_position[2] + pcb_holder_offset[2]
pcb_holder_thickness = pcb_holder_size[2]


def pcb_holder():
    logging.debug("pcb_holder()")
    shape = geometry_engine.box(*pcb_holder_size)
    shape = geometry_engine.translate(shape,
                                      (
                                          pcb_holder_position[0],
                                          pcb_holder_position[1] - pcb_holder_size[1] / 2,
                                          pcb_holder_thickness / 2,
                                      )
                                      )
    return shape


def wall_thinner():
    logging.debug("wall_thinner()")
    shape = geometry_engine.box(*wall_thinner_size)
    shape = geometry_engine.translate(shape,
                                      (
                                          pcb_holder_position[0],
                                          pcb_holder_position[1] - wall_thinner_size[1]/2,
                                          wall_thinner_size[2]/2 + pcb_holder_thickness,
                                      )
                                      )
    return shape


def trrs_hole():
    logging.debug("trrs_hole()")
    trrs_position = copy.deepcopy(pcb_mount_ref_position)
    trrs_position[0] = trrs_position[0] + trrs_offset[0]
    trrs_position[1] = trrs_position[1] + trrs_offset[1]
    trrs_position[2] = trrs_position[2] + trrs_offset[2]

    trrs_hole_size = [3, 20]

    shape = geometry_engine.cylinder(*trrs_hole_size)
    shape = geometry_engine.rotate(shape, [0, 90, 90])
    shape = geometry_engine.translate(shape,
                                      (
                                          trrs_position[0],
                                          trrs_position[1],
                                          trrs_hole_size[0] + pcb_holder_thickness,
                                      )
                                      )
    return shape


pcb_screw_position = copy.deepcopy(pcb_mount_ref_position)
pcb_screw_position[1] = pcb_screw_position[1] + pcb_screw_y_offset


def pcb_screw_hole():
    logging.debug("pcb_screw_hole()")
    holes = []
    hole = geometry_engine.cylinder(*pcb_screw_hole_size)
    hole = geometry_engine.translate(hole, pcb_screw_position)
    hole = geometry_engine.translate(hole, (0, 0, pcb_screw_hole_size[1]/2-.1))
    for offset in pcb_screw_x_offsets:
        holes.append(geometry_engine.translate(hole, (offset, 0, 0)))

    return holes


if oled_center_row is not None:
    base_pt1 = key_position(
        list(np.array([-mount_width/2, 0, 0]) + np.array([0, (mount_height / 2), 0])), 0, oled_center_row-1
    )
    base_pt2 = key_position(
        list(np.array([-mount_width/2, 0, 0]) + np.array([0, (mount_height / 2), 0])), 0, oled_center_row+1
    )
    base_pt0 = key_position(
        list(np.array([-mount_width / 2, 0, 0]) + np.array([0, (mount_height / 2), 0])), 0, oled_center_row
    )

    oled_mount_location_xyz = (np.array(base_pt1)+np.array(base_pt2))/2. + np.array(((-left_wall_x_offset/2), 0, 0)) + np.array(oled_translation_offset)
    oled_mount_location_xyz[2] = (oled_mount_location_xyz[2] + base_pt0[2])/2

    angle_x = math.atan2(base_pt1[2] - base_pt2[2], base_pt1[1] - base_pt2[1])
    angle_z = math.atan2(base_pt1[0] - base_pt2[0], base_pt1[1] - base_pt2[1])

    oled_mount_rotation_xyz = (math.degrees(angle_x), 0, -math.degrees(angle_z)) + np.array(oled_rotation_offset)


def generate_trackball(pos, rot):
    precut = trackball_cutout()
    precut = geometry_engine.rotate(precut, tb_socket_rotation_offset)
    precut = geometry_engine.translate(precut, tb_socket_translation_offset)
    precut = geometry_engine.rotate(precut, rot)
    precut = geometry_engine.translate(precut, pos)

    shape, cutout, sensor = trackball_socket()

    shape = geometry_engine.rotate(shape, tb_socket_rotation_offset)
    shape = geometry_engine.translate(shape, tb_socket_translation_offset)
    shape = geometry_engine.rotate(shape, rot)
    shape = geometry_engine.translate(shape, pos)

    cutout = geometry_engine.rotate(cutout, tb_socket_rotation_offset)
    cutout = geometry_engine.translate(cutout, tb_socket_translation_offset)
    # cutout = geometry_engine.rotate(cutout, tb_sensor_translation_offset)
    # cutout = geometry_engine.translate(cutout, tb_sensor_rotation_offset)
    cutout = geometry_engine.rotate(cutout, rot)
    cutout = geometry_engine.translate(cutout, pos)

    # Small adjustment due to line to line surface / minute numerical error issues
    # Creates small overlap to assist engines in geometry_engine.union function later
    sensor = geometry_engine.rotate(sensor, tb_socket_rotation_offset)
    sensor = geometry_engine.translate(sensor, tb_socket_translation_offset)
    # sensor = geometry_engine.rotate(sensor, tb_sensor_translation_offset)
    # sensor = geometry_engine.translate(sensor, tb_sensor_rotation_offset)
    sensor = geometry_engine.translate(sensor, (0, 0, .001))
    sensor = geometry_engine.rotate(sensor, rot)
    sensor = geometry_engine.translate(sensor, pos)

    ball = trackball_ball()
    ball = geometry_engine.rotate(ball, tb_socket_rotation_offset)
    ball = geometry_engine.translate(ball, tb_socket_translation_offset)
    ball = geometry_engine.rotate(ball, rot)
    ball = geometry_engine.translate(ball, pos)

    # return precut, shape, cutout, ball
    return precut, shape, cutout, sensor, ball


def generate_trackball_in_cluster():
    if thumb_style == 'TRACKBALL_ORBYL':
        pos, rot = tbjs_thumb_position_rotation()
    elif thumb_style == 'TRACKBALL_CJ':
        pos, rot = tbcj_thumb_position_rotation()
    return generate_trackball(pos, rot)


def tbiw_position_rotation():
    base_pt1 = key_position(
        list(np.array([-mount_width/2, 0, 0]) + np.array([0, (mount_height / 2), 0])),
        0, cornerrow - tbiw_ball_center_row - 1
    )
    base_pt2 = key_position(
        list(np.array([-mount_width/2, 0, 0]) + np.array([0, (mount_height / 2), 0])),
        0, cornerrow - tbiw_ball_center_row + 1
    )
    base_pt0 = key_position(
        list(np.array([-mount_width / 2, 0, 0]) + np.array([0, (mount_height / 2), 0])),
        0, cornerrow - tbiw_ball_center_row
    )

    left_wall_x_offset = tbiw_left_wall_x_offset_override

    tbiw_mount_location_xyz = (
        (np.array(base_pt1)+np.array(base_pt2))/2.
        + np.array(((-left_wall_x_offset/2), 0, 0))
        + np.array(tbiw_translational_offset)
    )

    # tbiw_mount_location_xyz[2] = (oled_translation_offset[2] + base_pt0[2])/2

    angle_x = math.atan2(base_pt1[2] - base_pt2[2], base_pt1[1] - base_pt2[1])
    angle_z = math.atan2(base_pt1[0] - base_pt2[0], base_pt1[1] - base_pt2[1])
    tbiw_mount_rotation_xyz = (math.degrees(angle_x), 0, math.degrees(angle_z)) + np.array(tbiw_rotation_offset)

    return tbiw_mount_location_xyz, tbiw_mount_rotation_xyz


def generate_trackball_in_wall():
    pos, rot = tbiw_position_rotation()
    return generate_trackball(pos, rot)


def oled_position_rotation(side='right'):
    _oled_center_row = None
    if trackball_in_wall and (side == ball_side or ball_side == 'both'):
        _oled_center_row = tbiw_oled_center_row
        _oled_translation_offset = tbiw_oled_translation_offset
        _oled_rotation_offset = tbiw_oled_rotation_offset

    elif oled_center_row is not None:
        _oled_center_row = oled_center_row
        _oled_translation_offset = oled_translation_offset
        _oled_rotation_offset = oled_rotation_offset

    if _oled_center_row is not None:
        base_pt1 = key_position(
            list(np.array([-mount_width/2, 0, 0]) + np.array([0, (mount_height / 2), 0])), 0, _oled_center_row-1
        )
        base_pt2 = key_position(
            list(np.array([-mount_width/2, 0, 0]) + np.array([0, (mount_height / 2), 0])), 0, _oled_center_row+1
        )
        base_pt0 = key_position(
            list(np.array([-mount_width / 2, 0, 0]) + np.array([0, (mount_height / 2), 0])), 0, _oled_center_row
        )

        if trackball_in_wall and (side == ball_side or ball_side == 'both'):
            _left_wall_x_offset = tbiw_left_wall_x_offset_override
        else:
            _left_wall_x_offset = left_wall_x_offset

        oled_mount_location_xyz = (np.array(base_pt1)+np.array(base_pt2))/2. + np.array(((-_left_wall_x_offset/2), 0, 0)) + np.array(_oled_translation_offset)
        oled_mount_location_xyz[2] = (oled_mount_location_xyz[2] + base_pt0[2])/2

        angle_x = math.atan2(base_pt1[2] - base_pt2[2], base_pt1[1] - base_pt2[1])
        angle_z = math.atan2(base_pt1[0] - base_pt2[0], base_pt1[1] - base_pt2[1])
        if trackball_in_wall and (side == ball_side or ball_side == 'both'):
            oled_mount_rotation_xyz = (0, math.degrees(angle_x), -90) + np.array(_oled_rotation_offset)
        else:
            oled_mount_rotation_xyz = (math.degrees(angle_x), 0, -math.degrees(angle_z)) + np.array(_oled_rotation_offset)

    return oled_mount_location_xyz, oled_mount_rotation_xyz


def oled_sliding_mount_frame(side='right'):
    mount_ext_width = oled_mount_width + 2 * oled_mount_rim
    mount_ext_height = (
        oled_mount_height + 2 * oled_edge_overlap_end
        + oled_edge_overlap_connector + oled_edge_overlap_clearance
        + 2 * oled_mount_rim
    )
    mount_ext_up_height = oled_mount_height + 2 * oled_mount_rim
    top_hole_start = -mount_ext_height / 2.0 + oled_mount_rim + oled_edge_overlap_end + oled_edge_overlap_connector
    top_hole_length = oled_mount_height

    hole = geometry_engine.box(mount_ext_width, mount_ext_up_height, oled_mount_cut_depth + .01)
    hole = geometry_engine.translate(hole, (0., top_hole_start + top_hole_length / 2, 0.))

    hole_down = geometry_engine.box(mount_ext_width, mount_ext_height, oled_mount_depth + oled_mount_cut_depth / 2)
    hole_down = geometry_engine.translate(hole_down, (0., 0., -oled_mount_cut_depth / 4))
    hole = geometry_engine.union([hole, hole_down])

    shape = geometry_engine.box(mount_ext_width, mount_ext_height, oled_mount_depth)

    conn_hole_start = -mount_ext_height / 2.0 + oled_mount_rim
    conn_hole_length = (
        oled_edge_overlap_end + oled_edge_overlap_connector
        + oled_edge_overlap_clearance + oled_thickness
    )
    conn_hole = geometry_engine.box(oled_mount_width, conn_hole_length + .01, oled_mount_depth)
    conn_hole = geometry_engine.translate(conn_hole, (
        0,
        conn_hole_start + conn_hole_length / 2,
        -oled_edge_overlap_thickness
    ))

    end_hole_length = (
        oled_edge_overlap_end + oled_edge_overlap_clearance
    )
    end_hole_start = mount_ext_height / 2.0 - oled_mount_rim - end_hole_length
    end_hole = geometry_engine.box(oled_mount_width, end_hole_length + .01, oled_mount_depth)
    end_hole = geometry_engine.translate(end_hole, (
        0,
        end_hole_start + end_hole_length / 2,
        -oled_edge_overlap_thickness
    ))

    top_hole_start = -mount_ext_height / 2.0 + oled_mount_rim + oled_edge_overlap_end + oled_edge_overlap_connector
    top_hole_length = oled_mount_height
    top_hole = geometry_engine.box(oled_mount_width, top_hole_length, oled_edge_overlap_thickness + oled_thickness - oled_edge_chamfer)
    top_hole = geometry_engine.translate(top_hole, (
        0,
        top_hole_start + top_hole_length / 2,
        (oled_mount_depth - oled_edge_overlap_thickness - oled_thickness - oled_edge_chamfer) / 2.0
    ))

    top_chamfer_1 = geometry_engine.box(
        oled_mount_width,
        top_hole_length,
        0.01
    )
    top_chamfer_2 = geometry_engine.box(
        oled_mount_width + 2 * oled_edge_chamfer,
        top_hole_length + 2 * oled_edge_chamfer,
        0.01
    )
    top_chamfer_1 = geometry_engine.translate(top_chamfer_1, (0, 0, -oled_edge_chamfer - .05))

    top_chamfer_1 = geometry_engine.convex_hull([top_chamfer_1, top_chamfer_2])

    top_chamfer_1 = geometry_engine.translate(top_chamfer_1, (
        0,
        top_hole_start + top_hole_length / 2,
        oled_mount_depth / 2.0 + .05
    ))

    top_hole = geometry_engine.union([top_hole, top_chamfer_1])

    shape = geometry_engine.difference(shape, [conn_hole, top_hole, end_hole])

    oled_mount_location_xyz, oled_mount_rotation_xyz = oled_position_rotation(side=side)

    shape = geometry_engine.rotate(shape, oled_mount_rotation_xyz)
    shape = geometry_engine.translate(shape,
                                      (
                                          oled_mount_location_xyz[0],
                                          oled_mount_location_xyz[1],
                                          oled_mount_location_xyz[2],
                                      )
                                      )

    hole = geometry_engine.rotate(hole, oled_mount_rotation_xyz)
    hole = geometry_engine.translate(hole,
                                     (
                                         oled_mount_location_xyz[0],
                                         oled_mount_location_xyz[1],
                                         oled_mount_location_xyz[2],
                                     )
                                     )
    return hole, shape


def oled_clip_mount_frame(side='right'):
    mount_ext_width = oled_mount_width + 2 * oled_mount_rim
    mount_ext_height = (
        oled_mount_height + 2 * oled_clip_thickness
        + 2 * oled_clip_undercut + 2 * oled_clip_overhang + 2 * oled_mount_rim
    )
    hole = geometry_engine.box(mount_ext_width, mount_ext_height, oled_mount_cut_depth + .01)

    shape = geometry_engine.box(mount_ext_width, mount_ext_height, oled_mount_depth)
    shape = geometry_engine.difference(shape, [geometry_engine.box(oled_mount_width, oled_mount_height, oled_mount_depth + .1)])

    clip_slot = geometry_engine.box(
        oled_clip_width + 2 * oled_clip_width_clearance,
        oled_mount_height + 2 * oled_clip_thickness + 2 * oled_clip_overhang,
        oled_mount_depth + .1
    )

    shape = geometry_engine.difference(shape, [clip_slot])

    clip_undercut = geometry_engine.box(
        oled_clip_width + 2 * oled_clip_width_clearance,
        oled_mount_height + 2 * oled_clip_thickness + 2 * oled_clip_overhang + 2 * oled_clip_undercut,
        oled_mount_depth + .1
    )

    clip_undercut = geometry_engine.translate(clip_undercut, (0., 0., oled_clip_undercut_thickness))
    shape = geometry_engine.difference(shape, [clip_undercut])

    plate = geometry_engine.box(
        oled_mount_width + .1,
        oled_mount_height - 2 * oled_mount_connector_hole,
        oled_mount_depth - oled_thickness
    )
    plate = geometry_engine.translate(plate, (0., 0., -oled_thickness / 2.0))
    shape = geometry_engine.union([shape, plate])

    oled_mount_location_xyz, oled_mount_rotation_xyz = oled_position_rotation(side=side)

    shape = geometry_engine.rotate(shape, oled_mount_rotation_xyz)
    shape = geometry_engine.translate(shape,
                                      (
                                          oled_mount_location_xyz[0],
                                          oled_mount_location_xyz[1],
                                          oled_mount_location_xyz[2],
                                      )
                                      )

    hole = geometry_engine.rotate(hole, oled_mount_rotation_xyz)
    hole = geometry_engine.translate(hole,
                                     (
                                         oled_mount_location_xyz[0],
                                         oled_mount_location_xyz[1],
                                         oled_mount_location_xyz[2],
                                     )
                                     )

    return hole, shape


def oled_clip():
    mount_ext_width = oled_mount_width + 2 * oled_mount_rim
    mount_ext_height = (
        oled_mount_height + 2 * oled_clip_thickness + 2 * oled_clip_overhang
        + 2 * oled_clip_undercut + 2 * oled_mount_rim
    )

    oled_leg_depth = oled_mount_depth + oled_clip_z_gap

    shape = geometry_engine.box(mount_ext_width - .1, mount_ext_height - .1, oled_mount_bezel_thickness)
    shape = geometry_engine.translate(shape, (0., 0., oled_mount_bezel_thickness / 2.))

    hole_1 = geometry_engine.box(
        oled_screen_width + 2 * oled_mount_bezel_chamfer,
        oled_screen_length + 2 * oled_mount_bezel_chamfer,
        .01
    )
    hole_2 = geometry_engine.box(oled_screen_width, oled_screen_length, 2.05 * oled_mount_bezel_thickness)
    hole = geometry_engine.convex_hull([hole_1, hole_2])

    shape = geometry_engine.difference(shape, [geometry_engine.translate(hole, (0., 0., oled_mount_bezel_thickness))])

    clip_leg = geometry_engine.box(oled_clip_width, oled_clip_thickness, oled_leg_depth)
    clip_leg = geometry_engine.translate(clip_leg, (
        0.,
        0.,
        # (oled_mount_height+2*oled_clip_overhang+oled_clip_thickness)/2,
        -oled_leg_depth / 2.
    ))

    latch_1 = geometry_engine.box(
        oled_clip_width,
        oled_clip_overhang + oled_clip_thickness,
        .01
    )
    latch_2 = geometry_engine.box(
        oled_clip_width,
        oled_clip_thickness / 2,
        oled_clip_extension
    )
    latch_2 = geometry_engine.translate(latch_2, (
        0.,
        -(-oled_clip_thickness / 2 + oled_clip_thickness + oled_clip_overhang) / 2,
        -oled_clip_extension / 2
    ))
    latch = geometry_engine.convex_hull([latch_1, latch_2])
    latch = geometry_engine.translate(latch, (
        0.,
        oled_clip_overhang / 2,
        -oled_leg_depth
    ))

    clip_leg = geometry_engine.union([clip_leg, latch])

    clip_leg = geometry_engine.translate(clip_leg, (
        0.,
        (oled_mount_height + 2 * oled_clip_overhang + oled_clip_thickness) / 2 - oled_clip_y_gap,
        0.
    ))

    shape = geometry_engine.union([shape, clip_leg, geometry_engine.mirror(clip_leg, 'XZ')])

    return shape


def oled_undercut_mount_frame(side='right'):
    mount_ext_width = oled_mount_width + 2 * oled_mount_rim
    mount_ext_height = oled_mount_height + 2 * oled_mount_rim
    hole = geometry_engine.box(mount_ext_width, mount_ext_height, oled_mount_cut_depth + .01)

    shape = geometry_engine.box(mount_ext_width, mount_ext_height, oled_mount_depth)
    shape = geometry_engine.difference(shape, [geometry_engine.box(oled_mount_width, oled_mount_height, oled_mount_depth + .1)])
    undercut = geometry_engine.box(
        oled_mount_width + 2 * oled_mount_undercut,
        oled_mount_height + 2 * oled_mount_undercut,
        oled_mount_depth)
    undercut = geometry_engine.translate(undercut, (0., 0., -oled_mount_undercut_thickness))
    shape = geometry_engine.difference(shape, [undercut])

    oled_mount_location_xyz, oled_mount_rotation_xyz = oled_position_rotation(side=side)

    shape = geometry_engine.rotate(shape, oled_mount_rotation_xyz)
    shape = geometry_engine.translate(shape, (
        oled_mount_location_xyz[0],
        oled_mount_location_xyz[1],
        oled_mount_location_xyz[2],
    )
    )

    hole = geometry_engine.rotate(hole, oled_mount_rotation_xyz)
    hole = geometry_engine.translate(hole, (
        oled_mount_location_xyz[0],
        oled_mount_location_xyz[1],
        oled_mount_location_xyz[2],
    )
    )

    return hole, shape


def teensy_holder():
    logging.debug("teensy_holder()")
    teensy_top_xy = key_position(wall_locate3(-1, 0), 0, centerrow - 1)
    teensy_bot_xy = key_position(wall_locate3(-1, 0), 0, centerrow + 1)
    teensy_holder_length = teensy_top_xy[1] - teensy_bot_xy[1]
    teensy_holder_offset = -teensy_holder_length / 2
    teensy_holder_top_offset = (teensy_holder_top_length / 2) - teensy_holder_length

    s1 = geometry_engine.box(3, teensy_holder_length, 6 + teensy_width)
    s1 = geometry_engine.translate(s1, [1.5, teensy_holder_offset, 0])

    s2 = geometry_engine.box(teensy_pcb_thickness, teensy_holder_length, 3)
    s2 = geometry_engine.translate(s2,
                                   (
                                       (teensy_pcb_thickness / 2) + 3,
                                       teensy_holder_offset,
                                       -1.5 - (teensy_width / 2),
                                   )
                                   )

    s3 = geometry_engine.box(teensy_pcb_thickness, teensy_holder_top_length, 3)
    s3 = geometry_engine.translate(s3,
                                   [
                                       (teensy_pcb_thickness / 2) + 3,
                                       teensy_holder_top_offset,
                                       1.5 + (teensy_width / 2),
                                   ]
                                   )

    s4 = geometry_engine.box(4, teensy_holder_top_length, 4)
    s4 = geometry_engine.translate(s4,
                                   [teensy_pcb_thickness + 5, teensy_holder_top_offset, 1 + (teensy_width / 2)]
                                   )

    shape = geometry_engine.union((s1, s2, s3, s4))

    shape = geometry_engine.translate(shape, [-teensy_holder_width, 0, 0])
    shape = geometry_engine.translate(shape, [-1.4, 0, 0])
    shape = geometry_engine.translate(shape,
                                      [teensy_top_xy[0], teensy_top_xy[1] - 1, (6 + teensy_width) / 2]
                                      )

    return shape


def screw_insert_shape(bottom_radius, top_radius, height):
    logging.debug("screw_insert_shape()")
    if bottom_radius == top_radius:
        base = geometry_engine.cylinder(radius=bottom_radius, height=height)
    else:
        base = geometry_engine.translate(cone(r1=bottom_radius, r2=top_radius, height=height), (0, 0, -height / 2))

    shape = geometry_engine.union((
        base,
        geometry_engine.translate(sphere(top_radius), (0, 0, height / 2)),
    ))
    return shape


def screw_insert(column, row, bottom_radius, top_radius, height, side='right'):
    logging.debug("screw_insert()")
    shift_right = column == lastcol
    shift_left = column == 0
    shift_up = (not (shift_right or shift_left)) and (row == 0)
    shift_down = (not (shift_right or shift_left)) and (row >= lastrow)

    if screws_offset == 'INSIDE':
        # logging.debug("Shift Inside")
        shift_left_adjust = wall_base_x_thickness
        shift_right_adjust = -wall_base_x_thickness/2
        shift_down_adjust = -wall_base_y_thickness/2
        shift_up_adjust = -wall_base_y_thickness/3

    elif screws_offset == 'OUTSIDE':
        logging.debug("Shift Outside")
        shift_left_adjust = 0
        shift_right_adjust = wall_base_x_thickness/2
        shift_down_adjust = wall_base_y_thickness*2/3
        shift_up_adjust = wall_base_y_thickness*2/3

    else:
        # logging.debug("Shift Origin")
        shift_left_adjust = 0
        shift_right_adjust = 0
        shift_down_adjust = 0
        shift_up_adjust = 0

    if shift_up:
        position = key_position(
            list(np.array(wall_locate2(0, 1)) + np.array([0, (mount_height / 2) + shift_up_adjust, 0])),
            column,
            row,
        )
    elif shift_down:
        position = key_position(
            list(np.array(wall_locate2(0, -1)) - np.array([0, (mount_height / 2) + shift_down_adjust, 0])),
            column,
            row,
        )
    elif shift_left:
        position = list(
            np.array(left_key_position(row, 0, side=side)) + np.array(wall_locate3(-1, 0)) + np.array((shift_left_adjust, 0, 0))
        )
    else:
        position = key_position(
            list(np.array(wall_locate2(1, 0)) + np.array([(mount_height / 2), 0, 0]) + np.array((shift_right_adjust, 0, 0))
                 ),
            column,
            row,
        )

    shape = screw_insert_shape(bottom_radius, top_radius, height)
    shape = geometry_engine.translate(shape, [position[0], position[1], height / 2])

    return shape


def thumb_screw_insert(bottom_radius, top_radius, height, offset=None, side='right'):
    shape = screw_insert_shape(bottom_radius, top_radius, height)
    shapes = []
    if offset is None:
        offset = 0.0

    origin = thumborigin()

    if ('TRACKBALL' in thumb_style) and not (side == ball_side or ball_side == 'both'):
        _thumb_style = other_thumb
    else:
        _thumb_style = thumb_style

    if _thumb_style == 'MINI':
        if separable_thumb:
            xypositions = copy.deepcopy(mini_separable_thumb_screw_xy_locations)
        else:
            xypositions = copy.deepcopy(mini_thumb_screw_xy_locations)

    elif _thumb_style == 'MINIDOX':
        if separable_thumb:
            xypositions = copy.deepcopy(minidox_separable_thumb_screw_xy_locations)
        else:
            xypositions = copy.deepcopy(minidox_thumb_screw_xy_locations)
        xypositions[0][1] = xypositions[0][1] - .4 * (minidox_Usize - 1) * sa_length

    elif _thumb_style == 'CARBONFET':
        if separable_thumb:
            xypositions = copy.deepcopy(carbonfet_separable_thumb_screw_xy_locations)
        else:
            xypositions = copy.deepcopy(carbonfet_thumb_screw_xy_locations)

    elif _thumb_style == 'TRACKBALL_ORBYL':
        if separable_thumb:
            xypositions = copy.deepcopy(orbyl_separable_thumb_screw_xy_locations)
        else:
            xypositions = copy.deepcopy(orbyl_thumb_screw_xy_locations)

    elif _thumb_style == 'TRACKBALL_CJ':
        if separable_thumb:
            xypositions = copy.deepcopy(tbcj_separable_thumb_screw_xy_locations)
        else:
            xypositions = copy.deepcopy(tbcj_thumb_screw_xy_locations)

    else:
        if separable_thumb:
            xypositions = copy.deepcopy(default_separable_thumb_screw_xy_locations)
        else:
            xypositions = copy.deepcopy(default_thumb_screw_xy_locations)

    for xyposition in xypositions:
        position = list(np.array(origin) + np.array([*xyposition, -origin[2]]))
        shapes.append(geometry_engine.translate(shape, [position[0], position[1], height / 2 + offset]))

    return shapes


def screw_insert_all_shapes(bottom_radius, top_radius, height, offset=0, side='right'):
    logging.debug("screw_insert_all_shapes()")
    shape = (
        geometry_engine.translate(screw_insert(0, 0, bottom_radius, top_radius, height, side=side), (0, 0, offset)),
        geometry_engine.translate(screw_insert(0, cornerrow, bottom_radius, top_radius, height, side=side), (0, left_wall_lower_y_offset, offset)),
        geometry_engine.translate(screw_insert(3, lastrow, bottom_radius, top_radius, height, side=side), (0, 0, offset)),
        geometry_engine.translate(screw_insert(3, 0, bottom_radius, top_radius, height, side=side), (0, 0, offset)),
        geometry_engine.translate(screw_insert(lastcol, 0, bottom_radius, top_radius, height, side=side), (0, 0, offset)),
        geometry_engine.translate(screw_insert(lastcol, cornerrow, bottom_radius, top_radius, height, side=side), (0, 0, offset)),
        # geometry_engine.translate(screw_insert_thumb(bottom_radius, top_radius, height), (0, 0, offset)),
    )

    return shape


def thumb_screw_insert_holes(side='right'):
    return thumb_screw_insert(
        screw_insert_bottom_radius, screw_insert_top_radius, screw_insert_height+.02, offset=-.01, side=side
    )


def thumb_screw_insert_outers(offset=0.0, side='right'):
    # screw_insert_bottom_radius + screw_insert_wall
    # screw_insert_top_radius + screw_insert_wall
    bottom_radius = screw_insert_outer_radius
    top_radius = screw_insert_outer_radius
    height = screw_insert_height + 1.5
    return thumb_screw_insert(bottom_radius, top_radius, height, offset=offset, side=side)


def screw_insert_holes(side='right'):
    return screw_insert_all_shapes(
        screw_insert_bottom_radius, screw_insert_top_radius, screw_insert_height+.02, offset=-.01, side=side
    )


def screw_insert_outers(offset=0.0, side='right'):
    # screw_insert_bottom_radius + screw_insert_wall
    # screw_insert_top_radius + screw_insert_wall
    bottom_radius = screw_insert_outer_radius
    top_radius = screw_insert_outer_radius
    height = screw_insert_height + 1.5
    return screw_insert_all_shapes(bottom_radius, top_radius, height, offset=offset, side=side)


def screw_insert_screw_holes(side='right'):
    return screw_insert_all_shapes(1.7, 1.7, 350, side=side)


def wire_post(direction, offset):
    logging.debug("wire_post()")
    s1 = geometry_engine.box(
        wire_post_diameter, wire_post_diameter, wire_post_height
    )
    s1 = geometry_engine.translate(s1, [0, -wire_post_diameter * 0.5 * direction, 0])

    s2 = geometry_engine.box(
        wire_post_diameter, wire_post_overhang, wire_post_diameter
    )
    s2 = geometry_engine.translate(s2,
                                   [0, -wire_post_overhang * 0.5 * direction, -wire_post_height / 2]
                                   )

    shape = geometry_engine.union((s1, s2))
    shape = geometry_engine.translate(shape, [0, -offset, (-wire_post_height / 2) + 3])
    shape = geometry_engine.rotate(shape, [-alpha / 2, 0, 0])
    shape = geometry_engine.translate(shape, (3, -mount_height / 2, 0))

    return shape


def wire_posts():
    logging.debug("wire_posts()")
    shape = default_thumb_ml_place(wire_post(1, 0).geometry_engine.translate([-5, 0, -2]))
    shape = geometry_engine.union([shape, default_thumb_ml_place(wire_post(-1, 6).geometry_engine.translate([0, 0, -2.5]))])
    shape = geometry_engine.union([shape, default_thumb_ml_place(wire_post(1, 0).geometry_engine.translate([5, 0, -2]))])

    for column in range(lastcol):
        for row in range(cornerrow):
            shape = geometry_engine.union([
                shape,
                key_place(wire_post(1, 0).geometry_engine.translate([-5, 0, 0]), column, row),
                key_place(wire_post(-1, 6).geometry_engine.translate([0, 0, 0]), column, row),
                key_place(wire_post(1, 0).geometry_engine.translate([5, 0, 0]), column, row),
            ])
    return shape


def model_side(side="right"):
    logging.debug("model_right()")
    #shape = add([key_holes(side=side)])
    shape = geometry_engine.union([key_holes(side=side)])
    if debug_exports:
        export_file(shape=shape, fname=path.join(r"..", "things", r"debug_key_plates"))
    connector_shape = connectors()
    shape = geometry_engine.union([shape, connector_shape])
    if debug_exports:
        export_file(shape=shape, fname=path.join(r"..", "things", r"debug_connector_shape"))
    walls_shape = case_walls(side=side, skeleton=skeletal)
    if debug_exports:
        export_file(shape=walls_shape, fname=path.join(r"..", "things", r"debug_walls_shape"))

    s2 = geometry_engine.union([walls_shape])
    s2 = geometry_engine.union([s2, *screw_insert_outers(side=side)])

    if controller_mount_type in ['RJ9_USB_TEENSY', 'USB_TEENSY']:
        s2 = geometry_engine.union([s2, teensy_holder()])

    if controller_mount_type in ['RJ9_USB_TEENSY', 'RJ9_USB_WALL', 'USB_WALL', 'USB_TEENSY']:
        s2 = geometry_engine.union([s2, usb_holder()])
        s2 = geometry_engine.difference(s2, [usb_holder_hole()])

    if controller_mount_type in ['RJ9_USB_TEENSY', 'RJ9_USB_WALL']:
        s2 = geometry_engine.difference(s2, [rj9_space()])

    if controller_mount_type in ['EXTERNAL']:
        s2 = geometry_engine.difference(s2, [external_mount_hole()])

    if controller_mount_type in ['PCB_MOUNT']:
        s2 = geometry_engine.difference(s2, [pcb_usb_hole()])
        s2 = geometry_engine.difference(s2, [trrs_hole()])
        s2 = geometry_engine.union([s2, pcb_holder()])
        s2 = geometry_engine.difference(s2, [wall_thinner()])
        s2 = geometry_engine.difference(s2, pcb_screw_hole())

    if controller_mount_type in [None, 'None']:
        0  # do nothing, only here to expressly state inaction.

    s2 = geometry_engine.difference(s2, [geometry_engine.union(screw_insert_holes(side=side))])
    shape = geometry_engine.union([shape, s2])

    if controller_mount_type in ['RJ9_USB_TEENSY', 'RJ9_USB_WALL']:
        shape = geometry_engine.union([shape, rj9_holder()])

    if oled_mount_type == "UNDERCUT":
        hole, frame = oled_undercut_mount_frame(side=side)
        shape = geometry_engine.difference(shape, [hole])
        shape = geometry_engine.union([shape, frame])

    elif oled_mount_type == "SLIDING":
        hole, frame = oled_sliding_mount_frame(side=side)
        shape = geometry_engine.difference(shape, [hole])
        shape = geometry_engine.union([shape, frame])

    elif oled_mount_type == "CLIP":
        hole, frame = oled_clip_mount_frame(side=side)
        shape = geometry_engine.difference(shape, [hole])
        shape = geometry_engine.union([shape, frame])

    if trackball_in_wall and (side == ball_side or ball_side == 'both') and separable_thumb:
        tbprecut, tb, tbcutout, sensor, ball = generate_trackball_in_wall()

        shape = geometry_engine.difference(shape, [tbprecut])
        # export_file(shape=shape, fname=path.join(save_path, config_name + r"_test_1"))
        shape = geometry_engine.union([shape, tb])
        # export_file(shape=shape, fname=path.join(save_path, config_name + r"_test_2"))
        shape = geometry_engine.difference(shape, [tbcutout])
        # export_file(shape=shape, fname=path.join(save_path, config_name + r"_test_3a"))
        # export_file(shape=add([shape, sensor]), fname=path.join(save_path, config_name + r"_test_3b"))
        shape = geometry_engine.union([shape, sensor])

        if show_caps:
            shape = add([shape, ball])

    if plate_pcb_clear:
        shape = geometry_engine.difference(shape, [plate_pcb_cutouts(side=side)])

    main_shape = shape

    # BUILD THUMB

    thumb_shape = thumb(side=side)
    if debug_exports:
        export_file(shape=thumb_shape, fname=path.join(r"..", "things", r"debug_thumb_shape"))
    thumb_connector_shape = thumb_connectors(side=side)
    if debug_exports:
        export_file(shape=thumb_connector_shape, fname=path.join(r"..", "things", r"debug_thumb_connector_shape"))

    thumb_wall_shape = thumb_walls(side=side, skeleton=skeletal)
    thumb_wall_shape = geometry_engine.union([thumb_wall_shape, *thumb_screw_insert_outers(side=side)])
    thumb_connection_shape = thumb_connection(side=side, skeleton=skeletal)

    if debug_exports:
        thumb_test = geometry_engine.union([thumb_shape, thumb_connector_shape, thumb_wall_shape, thumb_connection_shape])
        export_file(shape=thumb_test, fname=path.join(r"..", "things", r"debug_thumb_test_{}_shape".format(side)))

    thumb_section = geometry_engine.union([thumb_shape, thumb_connector_shape, thumb_wall_shape, thumb_connection_shape])
    thumb_section = geometry_engine.difference(thumb_section, [geometry_engine.union(thumb_screw_insert_holes(side=side))])

    has_trackball = False
    if ('TRACKBALL' in thumb_style) and (side == ball_side or ball_side == 'both'):
        logging.debug("Has Trackball")
        tbprecut, tb, tbcutout, sensor, ball = generate_trackball_in_cluster()
        has_trackball = True
        thumb_section = geometry_engine.difference(thumb_section, [tbprecut])
        if debug_exports:
            export_file(shape=thumb_section, fname=path.join(r"..", "things", r"debug_thumb_test_1_shape".format(side)))
        thumb_section = geometry_engine.union([thumb_section, tb])
        if debug_exports:
            export_file(shape=thumb_section, fname=path.join(r"..", "things", r"debug_thumb_test_2_shape".format(side)))
        thumb_section = geometry_engine.difference(thumb_section, [tbcutout])
        if debug_exports:
            export_file(shape=thumb_section, fname=path.join(r"..", "things", r"debug_thumb_test_3_shape".format(side)))
        thumb_section = geometry_engine.union([thumb_section, sensor])
        if debug_exports:
            export_file(shape=thumb_section, fname=path.join(r"..", "things", r"debug_thumb_test_4_shape".format(side)))

    if plate_pcb_clear:
        thumb_section = geometry_engine.difference(thumb_section, [thumb_pcb_plate_cutouts(side=side)])

    block = geometry_engine.box(350, 350, 40)
    block = geometry_engine.translate(block, (0, 0, -20))
    main_shape = geometry_engine.difference(main_shape, [block])
    thumb_section = geometry_engine.difference(thumb_section, [block])
    if debug_exports:
        export_file(shape=thumb_section, fname=path.join(r"..", "things", r"debug_thumb_test_5_shape".format(side)))

    if separable_thumb:
        thumb_section = geometry_engine.difference(thumb_section, [main_shape])
        if show_caps:
            thumb_section = add([thumb_section, thumbcaps(side=side)])
            if has_trackball:
                thumb_section = add([thumb_section, ball])
    else:
        main_shape = geometry_engine.union([main_shape, thumb_section])
        if debug_exports:
            export_file(shape=main_shape, fname=path.join(r"..", "things", r"debug_thumb_test_6_shape".format(side)))
        if show_caps:
            main_shape = add([main_shape, thumbcaps(side=side)])
            if has_trackball:
                main_shape = add([main_shape, ball])

        if trackball_in_wall and (side == ball_side or ball_side == 'both') and not separable_thumb:
            tbprecut, tb, tbcutout, sensor, ball = generate_trackball_in_wall()

            main_shape = geometry_engine.difference(main_shape, [tbprecut])
            # export_file(shape=shape, fname=path.join(save_path, config_name + r"_test_1"))
            main_shape = geometry_engine.union([main_shape, tb])
            # export_file(shape=shape, fname=path.join(save_path, config_name + r"_test_2"))
            main_shape = geometry_engine.difference(main_shape, [tbcutout])
            # export_file(shape=shape, fname=path.join(save_path, config_name + r"_test_3a"))
            # export_file(shape=add([shape, sensor]), fname=path.join(save_path, config_name + r"_test_3b"))
            main_shape = geometry_engine.union([main_shape, sensor])

            if show_caps:
                main_shape = add([main_shape, ball])

    if show_caps:
        main_shape = add([main_shape, caps()])

    if side == "left":
        main_shape = geometry_engine.mirror(main_shape, 'YZ')
        thumb_section = geometry_engine.mirror(thumb_section, 'YZ')

    return main_shape, thumb_section


# NEEDS TO BE SPECIAL FOR CADQUERY
# def baseplate(main_shape, base_shape, wedge_angle=None, side='right'):
def baseplate(wedge_angle=None, side='right'):
    if ENGINE == 'cadquery':
        # shape = mod_r

        thumb_shape = thumb(side=side)
        thumb_wall_shape = thumb_walls(side=side, skeleton=skeletal)
        thumb_wall_shape = geometry_engine.union([thumb_wall_shape, *thumb_screw_insert_outers(side=side)])
        thumb_connector_shape = thumb_connectors(side=side)
        thumb_connection_shape = thumb_connection(side=side, skeleton=skeletal)
        thumb_section = geometry_engine.union([thumb_shape, thumb_connector_shape, thumb_wall_shape, thumb_connection_shape])
        thumb_section = geometry_engine.difference(thumb_section, [geometry_engine.union(thumb_screw_insert_holes(side=side))])

        shape = geometry_engine.union([
            case_walls(side=side),
            *screw_insert_outers(side=side),
            thumb_section
        ])
        tool = screw_insert_all_shapes(screw_hole_diameter/2., screw_hole_diameter/2., 350, side=side)
        for item in tool:
            item = geometry_engine.translate(item, [0, 0, -10])
            shape = geometry_engine.difference(shape, [item])

        tool = thumb_screw_insert(screw_hole_diameter/2., screw_hole_diameter/2., 350, side=side)
        for item in tool:
            item = geometry_engine.translate(item, [0, 0, -10])
            shape = geometry_engine.difference(shape, [item])

        #shape = geometry_engine.union([main_shape, thumb_shape])

        shape = geometry_engine.translate(shape, (0, 0, -0.0001))

        square = cq.Workplane('XY').rect(1000, 1000)
        for wire in square.wires().objects:
            plane = cq.Workplane('XY').add(cq.Face.makeFromWires(wire))
        shape = intersect(shape, plane)

        outside = shape.vertices(cq.DirectionMinMaxSelector(cq.Vector(1, 0, 0), True)).objects[0]
        sizes = []
        max_val = 0
        inner_index = 0
        base_wires = shape.wires().objects
        for i_wire, wire in enumerate(base_wires):
            is_outside = False
            for vert in wire.Vertices():
                if vert.toTuple() == outside.toTuple():
                    outer_wire = wire
                    outer_index = i_wire
                    is_outside = True
                    sizes.append(0)
            if not is_outside:
                sizes.append(len(wire.Vertices()))
            if sizes[-1] > max_val:
                inner_index = i_wire
                max_val = sizes[-1]
        logging.debug(sizes)
        inner_wire = base_wires[inner_index]

        # inner_plate = cq.Workplane('XY').add(cq.Face.makeFromWires(inner_wire))
        if wedge_angle is not None:
            cq.Workplane('XY').add(cq.Solid.revolve(outerWire, innerWires, angleDegrees, axisStart, axisEnd))
        else:
            inner_shape = cq.Workplane('XY').add(cq.Solid.extrudeLinear(outerWire=inner_wire, innerWires=[], vecNormal=cq.Vector(0, 0, base_thickness)))
            inner_shape = geometry_engine.translate(inner_shape, (0, 0, -base_rim_thickness))

            holes = []
            for i in range(len(base_wires)):
                if i not in [inner_index, outer_index]:
                    holes.append(base_wires[i])
            cutout = [*holes, inner_wire]

            shape = cq.Workplane('XY').add(cq.Solid.extrudeLinear(outer_wire, cutout, cq.Vector(0, 0, base_rim_thickness)))
            hole_shapes = []
            for hole in holes:
                loc = hole.Center()
                hole_shapes.append(
                    geometry_engine.translate(
                        geometry_engine.cylinder(screw_cbore_diameter/2.0, screw_cbore_depth),
                        (loc.x, loc.y, 0)
                        # (loc.x, loc.y, screw_cbore_depth/2)
                    )
                )
            shape = geometry_engine.difference(shape, hole_shapes)
            shape = geometry_engine.translate(shape, (0, 0, -base_rim_thickness))
            shape = geometry_engine.union([shape, inner_shape])

        return shape
    else:

        shape = geometry_engine.union([
            case_walls(side=side),
            *screw_insert_outers(side=side),
            thumb_walls(side=side),
            *thumb_screw_insert_outers(side=side),
        ])

        tool = geometry_engine.translate(geometry_engine.union(screw_insert_screw_holes(side=side)), [0, 0, -10])
        base = geometry_engine.box(1000, 1000, .01)
        shape = shape - tool
        shape = intersect(shape, base)

        shape = geometry_engine.translate(shape, [0, 0, -0.001])

        return sl.projection(cut=True)(shape)


def run():

    mod_r, tmb_r = model_side(side="right")
    export_file(shape=mod_r, fname=path.join(save_path, config_name + r"_right"))
    export_file(shape=tmb_r, fname=path.join(save_path, config_name + r"_thumb_right"))

    #base = baseplate(mod_r, tmb_r, side='right')
    base = baseplate(side='right')
    export_file(shape=base, fname=path.join(save_path, config_name + r"_right_plate"))
    export_dxf(shape=base, fname=path.join(save_path, config_name + r"_right_plate"))

    if symmetry == "asymmetric":
        mod_l, tmb_l = model_side(side="left")
        export_file(shape=mod_l, fname=path.join(save_path, config_name + r"_left"))
        export_file(shape=tmb_l, fname=path.join(save_path, config_name + r"_thumb_left"))

        #base_l = geometry_engine.mirror(baseplate(mod_l, tmb_l, side='left'), 'YZ')
        base_l = geometry_engine.mirror(baseplate(side='left'), 'YZ')
        export_file(shape=base_l, fname=path.join(save_path, config_name + r"_left_plate"))
        export_dxf(shape=base_l, fname=path.join(save_path, config_name + r"_left_plate"))

    else:
        export_file(shape=geometry_engine.mirror(mod_r, 'YZ'), fname=path.join(save_path, config_name + r"_left"))

        lbase = geometry_engine.mirror(base, 'YZ')
        export_file(shape=lbase, fname=path.join(save_path, config_name + r"_left_plate"))
        export_dxf(shape=lbase, fname=path.join(save_path, config_name + r"_left_plate"))

    if oled_mount_type == 'UNDERCUT':
        export_file(shape=oled_undercut_mount_frame()[1], fname=path.join(save_path, config_name + r"_oled_undercut_test"))

    if oled_mount_type == 'SLIDING':
        export_file(shape=oled_sliding_mount_frame()[1], fname=path.join(save_path, config_name + r"_oled_sliding_test"))

    if oled_mount_type == 'CLIP':
        oled_mount_location_xyz = (0.0, 0.0, -oled_mount_depth / 2)
        oled_mount_rotation_xyz = (0.0, 0.0, 0.0)
        export_file(shape=oled_clip(), fname=path.join(save_path, config_name + r"_oled_clip"))
        export_file(shape=oled_clip_mount_frame()[1],
                    fname=path.join(save_path, config_name + r"_oled_clip_test"))
        export_file(shape=geometry_engine.union((oled_clip_mount_frame()[1], oled_clip())),
                    fname=path.join(save_path, config_name + r"_oled_clip_assy_test"))


# base = baseplate()
# export_file(shape=base, fname=path.join(save_path, config_name + r"_plate"))
if __name__ == '__main__':
    run()
