python examples/main.py exp=fix_dist_m3_DGN_cx dcd.sample=fix 'convex=[True, True]' asset_dir=examples/assets/object/DGN_5k/processed_data
python examples/main.py exp=fix_dist_m3_objv_cx dcd.sample=fix 'convex=[True, True]' asset_dir=examples/assets/object/objaverse_5k/processed_data
python examples/main.py exp=fix_dist_m3_DGN_cc dcd.sample=fix 'convex=[False, False]' asset_dir=examples/assets/object/DGN_5k/processed_data
python examples/main.py exp=fix_dist_m3_objv_cc dcd.sample=fix 'convex=[False, False]' asset_dir=examples/assets/object/objaverse_5k/processed_data
python examples/main.py exp=fix_dist_m0_DGN_cx dcd.sample=fix margin=0.0 'convex=[True, True]' asset_dir=examples/assets/object/DGN_5k/processed_data
python examples/main.py exp=fix_dist_m0_objv_cx dcd.sample=fix margin=0.0 'convex=[True, True]' asset_dir=examples/assets/object/objaverse_5k/processed_data
python examples/main.py exp=fix_dist_m0_DGN_cc dcd.sample=fix margin=0.0 'convex=[False, False]' asset_dir=examples/assets/object/DGN_5k/processed_data
python examples/main.py exp=fix_dist_m0_objv_cc dcd.sample=fix margin=0.0 'convex=[False, False]' asset_dir=examples/assets/object/objaverse_5k/processed_data
python examples/main.py exp=fix_dir_m3_DGN_cx dcd.sample=fix dcd.method=RS1Dir 'convex=[True, True]' asset_dir=examples/assets/object/DGN_5k/processed_data
python examples/main.py exp=fix_dir_m3_objv_cx dcd.sample=fix dcd.method=RS1Dir 'convex=[True, True]' asset_dir=examples/assets/object/objaverse_5k/processed_data
python examples/main.py exp=fix_dir_m3_DGN_cc dcd.sample=fix dcd.method=RS1Dir 'convex=[False, False]' asset_dir=examples/assets/object/DGN_5k/processed_data
python examples/main.py exp=fix_dir_m3_objv_cc dcd.sample=fix dcd.method=RS1Dir 'convex=[False, False]' asset_dir=examples/assets/object/objaverse_5k/processed_data
python examples/main.py exp=fix_dir_m0_DGN_cx dcd.sample=fix dcd.method=RS1Dir margin=0.0 'convex=[True, True]' asset_dir=examples/assets/object/DGN_5k/processed_data
python examples/main.py exp=fix_dir_m0_objv_cx dcd.sample=fix dcd.method=RS1Dir margin=0.0 'convex=[True, True]' asset_dir=examples/assets/object/objaverse_5k/processed_data
python examples/main.py exp=fix_dir_m0_DGN_cc dcd.sample=fix dcd.method=RS1Dir margin=0.0 'convex=[False, False]' asset_dir=examples/assets/object/DGN_5k/processed_data
python examples/main.py exp=fix_dir_m0_objv_cc dcd.sample=fix dcd.method=RS1Dir margin=0.0 'convex=[False, False]' asset_dir=examples/assets/object/objaverse_5k/processed_data
