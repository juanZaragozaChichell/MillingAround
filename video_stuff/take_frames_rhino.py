# -*- coding: utf-8 -*-
import rhinoscriptsyntax as rs
import scriptcontext as sc
import Rhino
import os

def get_material_index_by_name(material_name):
    """
    Search classic document materials (sc.doc.Materials) by name (case-insensitive).
    Returns index or -1 if not found.
    """
    if not material_name:
        return -1

    target = material_name.strip().lower()
    mats = sc.doc.Materials
    count = mats.Count
    for i in range(count):
        mat = mats[i]
        if mat is None:
            continue
        name = (mat.Name or "").strip().lower()
        if name == target:
            return i
    return -1


def get_render_material_by_name(material_name):
    """
    Search Render Materials (Panels: Materials) by name (case-insensitive).
    Returns the RenderMaterial object or None.
    """
    if not material_name:
        return None

    target = material_name.strip().lower()
    rms = sc.doc.RenderMaterials
    count = rms.Count
    for i in range(count):
        rm = rms[i]
        if rm is None:
            continue
        name = (rm.Name or "").strip().lower()
        if name == target:
            return rm
    return None


def move_objects_to_layer(objs, layer_name):
    """
    Ensure layer exists and move objects there.
    """
    if not rs.IsLayer(layer_name):
        rs.AddLayer(layer_name)
    for obj_id in objs:
        try:
            rs.ObjectLayer(obj_id, layer_name)
        except:
            print("  [WARN] Could not move object {} to layer '{}'".format(obj_id, layer_name))


def assign_material_to_objects(objs, material_name):
    """
    Try, in order:
      1) Classic material by name (sc.doc.Materials + rs.ObjectMaterialIndex)
      2) Render material by name (sc.doc.RenderMaterials, assign via RhinoCommon)
      3) Fallback: move objects to a layer with that name
    """
    # 1) Classic material table
    mat_index = get_material_index_by_name(material_name)
    if mat_index >= 0:
        print("  [INFO] Using classic material '{}' (index {})".format(material_name, mat_index))
        for obj_id in objs:
            try:
                rs.ObjectMaterialIndex(obj_id, mat_index)
                rs.ObjectMaterialSource(obj_id, 1)  # 0 = layer, 1 = object
            except:
                print("  [WARN] Could not assign classic material to object: {}".format(obj_id))
        return

    # 2) Render material table (Panels: Materials)
    rm = get_render_material_by_name(material_name)
    if rm is not None:
        print("  [INFO] Using render material '{}'".format(material_name))
        for obj_id in objs:
            rh_obj = sc.doc.Objects.Find(obj_id)
            if not rh_obj:
                continue
            attr = rh_obj.Attributes
            attr.RenderMaterial = rm
            attr.MaterialSource = Rhino.DocObjects.ObjectMaterialSource.MaterialFromObject
            sc.doc.Objects.ModifyAttributes(rh_obj, attr, True)
        sc.doc.Views.Redraw()
        return

    # 3) Fallback to layer
    print("  [INFO] Material '{}' not found; using layer fallback".format(material_name))
    move_objects_to_layer(objs, material_name)


def import_stl_with_material(frame_dir, filename, material_name):
    """
    Import STL from frame_dir/filename and assign material_name via assign_material_to_objects.
    Returns list of imported object ids.
    """
    path = os.path.join(frame_dir, filename)

    if not os.path.exists(path):
        print("  [WARN] File not found: {}".format(path))
        return []

    before = set(rs.AllObjects() or [])

    cmd = '_-Import "{}" _Enter'.format(path)
    rs.Command(cmd, False)

    after = set(rs.AllObjects() or [])
    new_objs = list(after - before)

    if not new_objs:
        print("  [WARN] No new objects detected for: {}".format(filename))
        return []

    assign_material_to_objects(new_objs, material_name)
    return new_objs


def main():
    base_dir = "./frames"

    if "Perspective" in rs.ViewNames():
        rs.CurrentView("Perspective")

    prev_imported_objs = []

    for i in range(450):
        frame_name = "frame_{}".format(i)
        frame_dir = os.path.join(base_dir, frame_name)

        print("\n=== Processing {} ===".format(frame_name))

        if not os.path.isdir(frame_dir):
            print("  [WARN] Frame directory not found: {}".format(frame_dir))
            continue

        # Delete only previously imported objects
        if prev_imported_objs:
            existing_to_delete = [oid for oid in prev_imported_objs if rs.IsObject(oid)]
            if existing_to_delete:
                rs.DeleteObjects(existing_to_delete)
            prev_imported_objs = []

        imported_this_frame = []

        # bottom_sphere.stl and top_sphere.stl -> "topbottom"
        imported_this_frame += import_stl_with_material(frame_dir, "bottom_sphere.stl", "topbottom")
        imported_this_frame += import_stl_with_material(frame_dir, "top_sphere.stl", "topbottom")

        # shank_axis_cylinder.stl -> "topbottom"
        imported_this_frame += import_stl_with_material(frame_dir, "shank_axis_cylinder.stl", "topbottom")

        # p_fp_spheres.stl -> "p_fp"
        imported_this_frame += import_stl_with_material(frame_dir, "p_fp_spheres.stl", "p_fp")

        # p_fp_line.stl -> "p_fp"
        imported_this_frame += import_stl_with_material(frame_dir, "p_fp_line.stl", "p_fp")

        # shank_cylinder.stl -> "Cylinder"
        imported_this_frame += import_stl_with_material(frame_dir, "shank_cylinder.stl", "Cylinder")

        # safe_spheres.stl -> "safeballs"
        imported_this_frame += import_stl_with_material(frame_dir, "safe_spheres.stl", "safeballs")

        # top_circle.stl and bottom_circle.stl -> "topbottom"
        imported_this_frame += import_stl_with_material(frame_dir, "top_circle.stl", "topbottom")
        imported_this_frame += import_stl_with_material(frame_dir, "bottom_circle.stl", "topbottom")

        prev_imported_objs = imported_this_frame

        # Deselect everything so capture shows the real shading
        rs.UnselectAllObjects()

        png_path = os.path.join(frame_dir, "{}.png".format(frame_name))
        print("  Capturing view to: {}".format(png_path))

        cmd_cap = '_-ViewCaptureToFile "{}" _Enter'.format(png_path)
        rs.Command(cmd_cap, True)

    print("\nAll frames processed.")


if __name__ == "__main__":
    main()