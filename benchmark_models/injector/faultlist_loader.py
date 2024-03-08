import csv

def convert_weights_coords_from_pt_to_tf(weight_coords):
    if len(weight_coords) == 2:
        out_features, in_features = weight_coords
        return (in_features, out_features)
    if len(weight_coords) == 4:
        out_channels, in_channels, filter_height, filter_width = weight_coords
        return (filter_height, filter_width, in_channels, out_channels)
    raise NotImplementedError(
        f"The rearrangement when weights have {len(weight_coords)} dimensions is not handled"
    )


def load_fault_list(fault_list_path: str, convert_faults_pt_to_tf=False):
    layer_list = []
    injections = []
    with open(fault_list_path) as f:
        cr = csv.reader(f)
        next(cr, None)  # Skip Header
        for row in cr:
            try:
                inj_id, layer_name, weight_pos, bit = row
                inj_id = int(inj_id)
                if layer_name not in layer_list:
                    layer_list.append(layer_name)

                weight_pos = [
                    int(coord.strip()) for coord in weight_pos.strip("()").split(",")
                ]
                if convert_faults_pt_to_tf:
                    weight_pos = convert_weights_coords_from_pt_to_tf(weight_pos)
                bit = int(bit)
                injections.append((inj_id, layer_name, weight_pos, bit))
            except Exception as e:
                print(f"Error happened processing the following row: {row}")
                raise e from None
        return layer_list, injections
