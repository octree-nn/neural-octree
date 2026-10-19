#!/usr/bin/env python3
"""Render one Objaverse GLB into the NeuOct feedforward data schema.

This file is executed by Blender, not by the project Python environment.
Normals stay in lossless EXR until the outer pipeline maps them to RGB PNG.
"""

import argparse
import glob
import json
import math
import os
import sys

import bpy
from mathutils import Vector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument(
        "--orbit-elevation-degrees",
        type=float,
        default=None,
        help="Use an evenly spaced horizontal orbit at this fixed elevation.",
    )
    parser.add_argument(
        "--color-only",
        action="store_true",
        help="Write only the diffuse-color pass and camera metadata.",
    )
    parser.add_argument(
        "--auto-frame",
        action="store_true",
        help="Choose the smallest safe camera radius from the normalized bounding sphere.",
    )
    parser.add_argument(
        "--frame-margin",
        type=float,
        default=1.08,
        help="Safety multiplier for --auto-frame (1.08 leaves an 8%% border margin).",
    )
    parser.add_argument("--fov-degrees", type=float, default=40.0)
    parser.add_argument(
        "--half-extent",
        type=float,
        default=1.0,
        help="Normalize the largest scene half-extent to this value (0.5 for local NeuOct data).",
    )
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--engine", choices=("CYCLES", "BLENDER_EEVEE"), default="CYCLES")
    parser.add_argument("--cycles-device", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use a one-bounce, no-denoise Cycles setup; geometry passes remain exact.",
    )
    parser.add_argument(
        "--single-layer",
        action="store_true",
        help="Use one material layer for color/depth/normal (faster for opaque textured assets).",
    )
    parser.add_argument(
        "--dump-mesh",
        default=None,
        help="Write the normalized (pipeline-frame) mesh to this .npz file.",
    )
    parser.add_argument(
        "--dump-mesh-only",
        action="store_true",
        help="Normalize, dump the mesh, and exit without rendering.",
    )
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    path = os.path.abspath(ARGS.object)
    extension = os.path.splitext(path)[1].lower()
    if extension in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=path, merge_vertices=True, import_shading="NORMALS")
    elif extension == ".obj":
        bpy.ops.import_scene.obj(filepath=path, axis_forward="Y", axis_up="Z")
    elif extension == ".ply":
        bpy.ops.import_mesh.ply(filepath=path)
    elif extension == ".stl":
        bpy.ops.import_mesh.stl(filepath=path, axis_forward="Y", axis_up="Z")
    else:
        raise ValueError(f"Unsupported mesh format: {extension}")
    for obj in list(bpy.context.scene.objects):
        if obj.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
        elif obj.type == "MESH" and not obj.data.materials and obj.data.vertex_colors.active:
            # PLY stores appearance in vertex colors rather than a material.
            material = bpy.data.materials.new("NeuOctVertexColor")
            material.use_nodes = True
            principled = material.node_tree.nodes.get("Principled BSDF")
            color = material.node_tree.nodes.new("ShaderNodeVertexColor")
            color.layer_name = obj.data.vertex_colors.active.name
            material.node_tree.links.new(color.outputs["Color"], principled.inputs["Base Color"])
            obj.data.materials.append(material)
    if not any(obj.type == "MESH" for obj in bpy.context.scene.objects):
        raise RuntimeError("Imported object contains no meshes")


def scene_bounds():
    lower = Vector((math.inf, math.inf, math.inf))
    upper = Vector((-math.inf, -math.inf, -math.inf))
    found = False
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        found = True
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            lower = Vector(tuple(min(a, b) for a, b in zip(lower, world)))
            upper = Vector(tuple(max(a, b) for a, b in zip(upper, world)))
    if not found:
        raise RuntimeError("No renderable meshes found")
    return lower, upper


def normalize_scene():
    roots = [obj for obj in bpy.context.scene.objects if obj.parent is None]
    parent = bpy.data.objects.new("NeuOct_Normalization", None)
    bpy.context.scene.collection.objects.link(parent)
    for obj in roots:
        if obj != parent:
            obj.parent = parent
    lower, upper = scene_bounds()
    extent = max(upper - lower)
    if not math.isfinite(extent) or extent <= 1.0e-8:
        raise RuntimeError("Object has a degenerate bounding box")
    # NeuOct's ShapeNet point clouds occupy [-1, 1]. Rays are subsequently
    # multiplied by ray_scale=0.5, so retaining the same source scale matters.
    if ARGS.half_extent <= 0:
        raise RuntimeError("--half-extent must be positive")
    scale = 2.0 * ARGS.half_extent / extent
    parent.scale = (scale, scale, scale)
    bpy.context.view_layer.update()
    lower, upper = scene_bounds()
    offset = -(lower + upper) * 0.5
    parent.matrix_world.translation += offset
    bpy.context.view_layer.update()
    lower, upper = scene_bounds()
    if max(abs(value) for value in (*lower, *upper)) > 1.001:
        raise RuntimeError("Scene normalization failed")
    return scale, offset, lower, upper


def _linear_to_srgb(x):
    """Blender float buffers are scene-linear; the pipeline's albedo PNGs are sRGB."""
    import numpy as np

    x = np.clip(x, 0.0, 1.0)
    low = x * 12.92
    high = 1.055 * np.power(np.maximum(x, 1e-8), 1.0 / 2.4) - 0.055
    return np.where(x <= 0.0031308, low, high)


def _base_color_source(material):
    """Resolve a Blender material's Base Color into an image (+factor) or a colour.

    Returns ``("image", image, factor_rgb)``, ``("rgb", rgb)`` or ``None``.  The
    glTF importer wires ``Image Texture -> (Mix multiply with baseColorFactor) ->
    Base Color``, so the search walks *backwards* from the Principled BSDF's Base
    Color socket and keeps the first image texture it meets, plus any constant
    RGB node on the way (the factor).
    """
    if material is None or not getattr(material, "use_nodes", False):
        return None
    tree = material.node_tree
    principled = next((n for n in tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if principled is None:
        return None
    socket = principled.inputs.get("Base Color")
    if socket is None:
        return None
    default = tuple(float(c) for c in socket.default_value)[:3]
    if not socket.is_linked:
        return ("rgb", default)
    stack = [socket.links[0].from_node]
    image = None
    factor = None
    seen = set()
    while stack:
        node = stack.pop()
        if node.name in seen:
            continue
        seen.add(node.name)
        if node.type == "TEX_IMAGE" and getattr(node, "image", None) is not None:
            if image is None and node.image.size[0] * node.image.size[1] > 0:
                image = node.image
            continue
        if node.type == "RGB" and factor is None:
            factor = tuple(float(c) for c in node.outputs[0].default_value)[:3]
        for item in node.inputs:
            for link in item.links:
                stack.append(link.from_node)
    if image is not None:
        return ("image", image, factor if factor is not None else (1.0, 1.0, 1.0))
    if factor is not None:
        return ("rgb", factor)
    return ("rgb", default)


def _socket_image_channel(socket):
    """Follow a node socket back to ``(image, channel_index)`` if one feeds it.

    Handles the glTF ORM wiring ``Image Texture -> Separate RGB -> Metallic`` by
    carrying the channel implied by the Separate-RGB output socket name.
    """
    seen: set[str] = set()
    stack = [(socket, 0)]
    while stack:
        current, channel = stack.pop()
        if not current.is_linked:
            continue
        link = current.links[0]
        node = link.from_node
        name = link.from_socket.name
        if node.type == "TEX_IMAGE" and getattr(node, "image", None) is not None:
            if node.image.size[0] * node.image.size[1] == 0:
                continue
            return node.image, {"Red": 0, "Green": 1, "Blue": 2, "Alpha": 3}.get(name, 0)
        if node.name in seen:
            continue
        seen.add(node.name)
        if node.type == "SEPRGB":
            stack.append((node.inputs["Image"], {"R": 0, "G": 1, "B": 2}.get(name, channel)))
            continue
        if node.type == "RGBTOBW":
            stack.append((node.inputs["Color"], 0))
            continue
        for item in node.inputs:
            for item_link in item.links:
                stack.append((item_link.from_node.outputs[item_link.from_socket.name], channel))
    return None, None


def _metallic_sampler(np, material):
    """``(sampler, constant)`` for the material's Metallic input, or ``(None, 0.0)``.

    Cycles' *Diffuse Color* pass -- which is what the pipeline's albedo PNGs hold --
    is ``base_color * (1 - metallic)``, so a texture-only colour is wrong wherever
    the ORM map marks metal.  ``sampler(uv)`` returns metallic in ``[0, 1]``.
    """
    if material is None or not getattr(material, "use_nodes", False):
        return None, 0.0
    principled = next((n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if principled is None:
        return None, 0.0
    socket = principled.inputs.get("Metallic")
    if socket is None:
        return None, 0.0
    if not socket.is_linked:
        return None, float(socket.default_value)
    image, channel = _socket_image_channel(socket)
    if image is None:
        return None, 0.0
    width, height = int(image.size[0]), int(image.size[1])
    buf = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(buf)
    plane = np.ascontiguousarray(buf.reshape(height, width, 4)[..., channel])  # Non-Color: no decode

    def sample(uv_coords):
        u = np.mod(uv_coords[:, 0], 1.0)
        v = np.mod(uv_coords[:, 1], 1.0)
        xi = np.clip((u * width).astype(np.int64), 0, width - 1)
        yi = np.clip((v * height).astype(np.int64), 0, height - 1)
        return plane[yi, xi]

    return sample, 0.0


def _image_sampler(np, image, factor):
    """Nearest-neighbour sRGB sampler over a Blender image, in [0, 1] like albedo/255."""
    width, height = int(image.size[0]), int(image.size[1])
    buf = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(buf)
    # Blender stores rows bottom-up.  Measured: `Image.pixels` for an 8-bit sRGB
    # image returns the *sRGB-encoded* byte values divided by 255, i.e. exactly what
    # the pipeline's albedo PNGs hold -- do NOT apply a linear->sRGB transfer here.
    pixels = buf.reshape(height, width, 4)
    rgb = np.ascontiguousarray(pixels[..., :3]) * np.asarray(factor, dtype=np.float32)

    def sample(uv):
        u = np.mod(uv[:, 0], 1.0)
        v = np.mod(uv[:, 1], 1.0)
        xi = np.clip((u * width).astype(np.int64), 0, width - 1)
        yi = np.clip((v * height).astype(np.int64), 0, height - 1)
        return rgb[yi, xi]

    return sample


def _vertex_colors_from_texture(np, mesh, material_slots):
    """Per-vertex base colour sampled from each material's texture through the UVs.

    Returns ``(colors, covered)`` where ``covered`` is the per-vertex boolean mask
    that actually got a textured sample.  Fully vectorised: no Python loop over
    loops, because Objaverse meshes reach tens of millions of loops.
    """
    uv_layer = mesh.uv_layers.active if len(mesh.uv_layers) else None
    if uv_layer is None or len(mesh.polygons) == 0:
        return None, None

    loop_count = len(mesh.loops)
    vertex_count = len(mesh.vertices)
    uvs = np.empty(loop_count * 2, dtype=np.float32)
    uv_layer.data.foreach_get("uv", uvs)
    uvs = uvs.reshape(loop_count, 2)

    vertex_index = np.empty(loop_count, dtype=np.int64)
    mesh.loops.foreach_get("vertex_index", vertex_index)

    polygons = len(mesh.polygons)
    material_index = np.empty(polygons, dtype=np.int32)
    mesh.polygons.foreach_get("material_index", material_index)
    loop_total = np.empty(polygons, dtype=np.int32)
    mesh.polygons.foreach_get("loop_total", loop_total)
    loop_material = np.repeat(material_index, loop_total)

    accum = np.zeros((vertex_count, 3), dtype=np.float64)
    counts = np.zeros(vertex_count, dtype=np.float64)
    samplers = {}
    for slot_index, slot in enumerate(material_slots):
        material = slot.material if slot else None
        source = _base_color_source(material)
        if source is None:
            continue
        if source[0] == "image":
            base = _image_sampler(np, source[1], source[2])
        else:
            constant = np.asarray(source[1], dtype=np.float32)
            base = lambda uv, c=constant: np.broadcast_to(c, (uv.shape[0], 3))
        samplers[slot_index] = (base, _metallic_sampler(np, material))
    if not samplers:
        return None, None

    for slot_index, (sampler, metallic) in samplers.items():
        selected = loop_material == slot_index
        if not selected.any():
            continue
        uv_selected = uvs[selected]
        color = sampler(uv_selected)
        metal = metallic[0](uv_selected) if metallic[0] is not None else metallic[1]
        # reproduce Cycles' Diffuse Color pass (what the pipeline's albedo PNGs hold)
        metal = np.asarray(metal, dtype=np.float32)
        color = color * (1.0 - metal) if metal.ndim == 0 else color * (1.0 - metal)[:, None]
        index = vertex_index[selected]
        for channel in range(3):
            accum[:, channel] += np.bincount(index, weights=color[:, channel],
                                             minlength=vertex_count)
        counts += np.bincount(index, minlength=vertex_count)
    covered = counts > 0
    if not covered.any():
        return None, None
    safe = np.maximum(counts, 1.0)[:, None]
    return (accum / safe).astype(np.float32), covered


def dump_normalized_mesh(path):
    """Write the scene's triangles in the *pipeline frame* (post-normalization).

    Vertices are world-space coordinates after normalize_scene(), i.e. exactly the
    frame the depth maps / point cloud live in, so no post-hoc registration is
    needed.  Each mesh object is evaluated (modifiers applied) and triangulated.

    ``colors`` is the per-vertex albedo, in the same convention as the pipeline's
    point clouds (sRGB-encoded, in ``[0, 1]``).  Priority:

      1. **material Base Color sampled through the mesh UVs** -- the mesh's own
         texture, which is the only colour source that is complete where the
         8-view point cloud never saw the surface;
      2. the mesh's ``color_attributes`` (vertex colours), for untextured assets;
      3. zeros, which the occupancy build then fills from the point cloud.
    """
    import numpy as np

    depsgraph = bpy.context.evaluated_depsgraph_get()
    all_vertices = []
    all_faces = []
    all_colors = []
    stats = {"texture": 0, "vertex_color": 0, "none": 0}
    offset = 0
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        mesh.calc_loop_triangles()
        if len(mesh.vertices) == 0 or len(mesh.loop_triangles) == 0:
            evaluated.to_mesh_clear()
            continue
        matrix = evaluated.matrix_world
        vertices = np.array([list(matrix @ v.co) for v in mesh.vertices], dtype=np.float32)
        faces = np.array([list(t.vertices) for t in mesh.loop_triangles], dtype=np.int64)

        try:
            colors, covered = _vertex_colors_from_texture(np, mesh, obj.material_slots)
        except Exception as error:  # noqa: BLE001 - geometry matters more than colour
            print(f"[WARN] texture sampling failed on {obj.name}: {error!r}", flush=True)
            colors, covered = None, None

        if colors is None:
            # Blender 3.0 has no Mesh.color_attributes (added in 3.2); it only exposes
            # the legacy loop-domain Mesh.vertex_colors.  Without this fallback the
            # colour lookup silently produced all-zero vertex colours on every asset.
            # Everything here is vectorised and wrapped: `mesh.loops` yields bpy
            # objects whose `vertex_index` is a bpy_prop, not an int, so a Python-level
            # `corner[vertex_index] +=` raised IndexError and killed the whole dump
            # (423 of 32,761 objects) -- a colour detail must never cost us geometry.
            try:
                attribute = None
                domain = None
                if hasattr(mesh, "color_attributes"):
                    attribute = mesh.color_attributes.active_color
                    domain = getattr(attribute, "domain", None)
                elif len(mesh.vertex_colors):
                    attribute = mesh.vertex_colors.active
                    domain = "CORNER"
                loop_count = len(mesh.loops)
                vertex_count = len(mesh.vertices)
                if attribute is not None and domain == "CORNER" and loop_count:
                    per_loop = np.zeros(loop_count * 4, dtype=np.float32)
                    attribute.data.foreach_get("color", per_loop)
                    per_loop = per_loop.reshape(loop_count, 4)
                    loop_vertex = np.empty(loop_count, dtype=np.int64)
                    mesh.loops.foreach_get("vertex_index", loop_vertex)
                    counts = np.bincount(loop_vertex, minlength=vertex_count).astype(np.float32)
                    corner = np.zeros((vertex_count, 4), dtype=np.float32)
                    for channel in range(4):
                        corner[:, channel] = np.bincount(
                            loop_vertex, weights=per_loop[:, channel],
                            minlength=vertex_count).astype(np.float32)
                    counts[counts == 0] = 1.0
                    colors = (corner / counts[:, None])[:, :3]
                elif attribute is not None and domain == "POINT":
                    colors = np.zeros((vertex_count, 4), dtype=np.float32)
                    attribute.data.foreach_get("color", colors.reshape(-1))
                    colors = colors[:, :3]
            except Exception as error:  # noqa: BLE001
                print(f"[WARN] vertex-colour fallback failed on {obj.name}: {error!r}",
                      flush=True)
                colors = None

        if colors is None:
            stats["none"] += 1
            colors = np.zeros((len(vertices), 3), np.float32)
        elif covered is not None:
            stats["texture"] += 1
        else:
            stats["vertex_color"] += 1

        all_vertices.append(vertices)
        all_faces.append(faces + offset)
        all_colors.append(colors)
        offset += len(vertices)
        evaluated.to_mesh_clear()

    if not all_vertices:
        raise RuntimeError("No mesh geometry to dump")
    vertices = np.concatenate(all_vertices, axis=0)
    faces = np.concatenate(all_faces, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(path, vertices=vertices, faces=faces, colors=colors)
    nonblack = float((np.abs(colors).sum(axis=1) > 1e-6).mean()) if len(colors) else 0.0
    print(
        "[INFO] dumped pipeline-frame mesh: {} vertices={} faces={} "
        "bbox=[{:.4f},{:.4f}] color_source={} nonblack_vertices={:.1%} "
        "color_range=[{:.3f},{:.3f}]".format(
            path, len(vertices), len(faces),
            float(np.abs(vertices).max()), float(-np.abs(vertices).max()),
            stats, nonblack, float(colors.min()), float(colors.max()),
        ),
        flush=True,
    )
    return vertices, faces


def configure_cycles(scene, samples):
    scene.cycles.samples = samples
    scene.cycles.use_denoising = samples > 1 and not ARGS.fast
    scene.cycles.device = ARGS.cycles_device
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 2
    scene.cycles.transmission_bounces = 3
    scene.cycles.transparent_max_bounces = 4
    if ARGS.fast:
        scene.cycles.max_bounces = 1
        scene.cycles.diffuse_bounces = 0
        scene.cycles.glossy_bounces = 0
        scene.cycles.transmission_bounces = 0
        scene.cycles.transparent_max_bounces = 1
    if ARGS.cycles_device == "CPU":
        return
    preferences = bpy.context.preferences.addons["cycles"].preferences
    try:
        preferences.compute_device_type = "CUDA"
        preferences.get_devices()
        for device in preferences.devices:
            device.use = device.type != "CPU"
    except Exception as error:
        print("[WARN] Could not configure CUDA devices: {}".format(error), flush=True)


def configure_render():
    scene = bpy.context.scene
    scene.render.engine = ARGS.engine
    scene.render.resolution_x = ARGS.resolution
    scene.render.resolution_y = ARGS.resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 30
    scene.render.use_file_extension = True
    if ARGS.engine == "CYCLES":
        configure_cycles(scene, ARGS.samples)
    else:
        scene.eevee.taa_render_samples = max(8, ARGS.samples)
        scene.eevee.use_gtao = True
        scene.eevee.gtao_distance = 3
        scene.eevee.gtao_factor = 1.25
    color_layer = scene.view_layers[0]
    color_layer.name = "NeuOctColor"
    color_layer.use_pass_z = True
    color_layer.use_pass_normal = True
    color_layer.use_pass_diffuse_color = True
    return scene, color_layer


def add_geometry_view_layer(scene):
    """Render material-independent depth/normals for transparent assets."""
    material = bpy.data.materials.new("NeuOctOpaqueGeometry")
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled is not None:
        principled.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
        principled.inputs["Roughness"].default_value = 1.0
        principled.inputs["Alpha"].default_value = 1.0

    layer = scene.view_layers.new("NeuOctGeometry")
    layer.material_override = material
    layer.use_pass_z = True
    layer.use_pass_normal = True
    layer.samples = 1
    return layer


def configure_compositor(scene, color_layer, geometry_layer):
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    nodes.clear()
    color_layers = nodes.new("CompositorNodeRLayers")
    color_layers.layer = color_layer.name
    depth_map = None
    depth_out = None
    normal_out = None
    if not ARGS.color_only:
        geometry_layers = nodes.new("CompositorNodeRLayers")
        geometry_layers.layer = geometry_layer.name

        depth_map = nodes.new("CompositorNodeMapRange")
        depth_map.use_clamp = True
        scene.node_tree.links.new(geometry_layers.outputs["Depth"], depth_map.inputs[0])
        depth_out = nodes.new("CompositorNodeOutputFile")
        depth_out.base_path = ""
        depth_out.format.file_format = "PNG"
        depth_out.format.color_mode = "BW"
        depth_out.format.color_depth = "16"
        scene.node_tree.links.new(depth_map.outputs[0], depth_out.inputs[0])

        normal_out = nodes.new("CompositorNodeOutputFile")
        normal_out.base_path = ""
        normal_out.format.file_format = "OPEN_EXR"
        normal_out.format.color_mode = "RGB"
        normal_out.format.color_depth = "32"
        scene.node_tree.links.new(geometry_layers.outputs["Normal"], normal_out.inputs[0])

    alpha = nodes.new("CompositorNodeSetAlpha")
    scene.node_tree.links.new(color_layers.outputs["DiffCol"], alpha.inputs["Image"])
    scene.node_tree.links.new(color_layers.outputs["Alpha"], alpha.inputs["Alpha"])
    albedo_out = nodes.new("CompositorNodeOutputFile")
    albedo_out.base_path = ""
    albedo_out.format.file_format = "PNG"
    albedo_out.format.color_mode = "RGBA"
    albedo_out.format.color_depth = "8"
    scene.node_tree.links.new(alpha.outputs["Image"], albedo_out.inputs[0])
    return depth_map, depth_out, normal_out, albedo_out


def add_camera(scene):
    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera_data.sensor_width = 32
    camera_data.sensor_height = 32
    camera_data.lens = 16.0 / math.tan(math.radians(ARGS.fov_degrees) * 0.5)
    camera_data.clip_start = 0.1
    camera_data.clip_end = 10.0
    constraint = camera.constraints.new(type="TRACK_TO")
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"
    target = bpy.data.objects.new("CameraTarget", None)
    scene.collection.objects.link(target)
    constraint.target = target
    return camera


def add_lighting(scene):
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.8, 0.8, 0.8, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.7
    for name, location, energy, size in (
        ("Key", (4, -4, 6), 750, 5),
        ("Fill", (-4, -2, 3), 500, 4),
        ("Rim", (1, 4, -2), 350, 3),
    ):
        data = bpy.data.lights.new(name, type="AREA")
        data.energy = energy
        data.size = size
        light = bpy.data.objects.new(name, data)
        light.location = location
        scene.collection.objects.link(light)


def matrix_list(obj):
    return [[float(value) for value in row] for row in obj.matrix_world]


def view_angles(index, count):
    if ARGS.orbit_elevation_degrees is not None:
        yaw = index * (2.0 * math.pi / count)
        return yaw, math.radians(ARGS.orbit_elevation_degrees)
    golden = math.pi * (3.0 - math.sqrt(5.0))
    z = 1.0 - 2.0 * (index + 0.5) / count
    pitch = math.asin(z)
    yaw = (index * golden) % (2.0 * math.pi)
    return yaw, pitch


def move_compositor_file(source_prefix, destination):
    matches = glob.glob(source_prefix + "*." + destination.rsplit(".", 1)[-1])
    if len(matches) != 1:
        raise RuntimeError("Expected one compositor output for {}, got {}".format(source_prefix, matches))
    os.replace(matches[0], destination)


def render_views(
    scene, camera, depth_map, depth_out, normal_out, albedo_out, metadata, camera_radius
):
    fov = math.radians(ARGS.fov_degrees)
    depth_min = camera_radius - math.sqrt(3.0)
    depth_max = camera_radius + math.sqrt(3.0)
    if depth_map is not None:
        depth_map.inputs[1].default_value = depth_min
        depth_map.inputs[2].default_value = depth_max
        depth_map.inputs[3].default_value = 0
        depth_map.inputs[4].default_value = 1
    for index in range(ARGS.views):
        yaw, pitch = view_angles(index, ARGS.views)
        camera.location = (
            camera_radius * math.cos(yaw) * math.cos(pitch),
            camera_radius * math.sin(yaw) * math.cos(pitch),
            camera_radius * math.sin(pitch),
        )
        bpy.context.view_layer.update()
        stem = os.path.join(ARGS.output, "{:03d}".format(index))
        scene.render.filepath = stem + ".png"
        if depth_out is not None:
            depth_out.file_slots[0].path = stem + "_depth_tmp_"
        if normal_out is not None:
            normal_out.file_slots[0].path = stem + "_normal_tmp_"
        albedo_out.file_slots[0].path = stem + "_albedo_tmp_"
        bpy.ops.render.render(write_still=not ARGS.color_only)
        if depth_out is not None:
            move_compositor_file(stem + "_depth_tmp_", stem + "_depth.png")
        if normal_out is not None:
            move_compositor_file(stem + "_normal_tmp_", stem + "_normal.exr")
        move_compositor_file(stem + "_albedo_tmp_", stem + "_albedo.png")
        metadata["frames"].append(
            {
                "file_path": "{:03d}.png".format(index),
                "camera_angle_x": fov,
                "transform_matrix": matrix_list(camera),
                "depth": {"min": depth_min, "max": depth_max},
                "yaw": yaw,
                "pitch": pitch,
                "radius": camera_radius,
            }
        )
        print("[INFO] rendered view {}/{}".format(index + 1, ARGS.views), flush=True)


def main():
    os.makedirs(ARGS.output, exist_ok=True)
    clear_scene()
    scale, offset, lower, upper = normalize_scene()
    if ARGS.dump_mesh:
        dump_normalized_mesh(ARGS.dump_mesh)
    if ARGS.dump_mesh_only:
        print("[INFO] --dump-mesh-only: skipping render", flush=True)
        return
    scene, color_layer = configure_render()
    geometry_layer = color_layer if ARGS.single_layer else add_geometry_view_layer(scene)
    compositor = configure_compositor(scene, color_layer, geometry_layer)
    camera = add_camera(scene)
    add_lighting(scene)
    metadata = {
        "schema_version": 1,
        "source": os.path.basename(ARGS.object),
        "aabb": [[float(v) for v in lower], [float(v) for v in upper]],
        "normalization": {
            "scale": float(scale),
            "offset": [float(value) for value in offset],
        },
        "frames": [],
    }
    if ARGS.frame_margin <= 1.0:
        raise RuntimeError("--frame-margin must be greater than 1")
    sphere_radius = 0.5 * (upper - lower).length
    if ARGS.auto_frame:
        fov_half = math.radians(ARGS.fov_degrees) * 0.5
        if fov_half <= 0.0 or fov_half >= math.pi * 0.5:
            raise RuntimeError("--fov-degrees must be in (0, 180) for auto framing")
        camera_radius = ARGS.frame_margin * sphere_radius / math.sin(fov_half)
    else:
        camera_radius = ARGS.radius
    if camera_radius <= sphere_radius:
        raise RuntimeError("Camera radius must exceed the scene bounding-sphere radius")
    metadata["camera"] = {
        "radius": float(camera_radius),
        "fov_degrees": float(ARGS.fov_degrees),
        "auto_frame": bool(ARGS.auto_frame),
        "frame_margin": float(ARGS.frame_margin),
        "bounding_sphere_radius": float(sphere_radius),
    }
    metadata["view_mode"] = (
        "horizontal_orbit" if ARGS.orbit_elevation_degrees is not None else "fibonacci"
    )
    metadata["orbit_elevation_degrees"] = ARGS.orbit_elevation_degrees
    render_views(scene, camera, *compositor, metadata, camera_radius)
    with open(os.path.join(ARGS.output, "transforms.json"), "w") as handle:
        json.dump(metadata, handle, indent=2)


ARGS = parse_args()
main()
