import argparse
import json
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, "optimizer"))

from optimization import customize, create_trig_features

def main():

    parser = argparse.ArgumentParser(description='Run optimization with parameters')
    parser.add_argument('--params', required=True, help='Path to JSON file with parameters')
    args = parser.parse_args()
    
    try:

        with open(args.params, 'r') as f:
            params = json.load(f)
        

        input_params = {
            'naturalAngle': params.get('naturalAngle', 22.0),
            'ptorqueExtend': params.get('ptorqueExtend', 30.0),
            'atorqueBend': params.get('atorqueBend', 50.0),
            'atorqueExtend': params.get('atorqueExtend', 20.0)
        }
        

        output_dir = params.get('output_dir', "optimization_results")
        os.makedirs(output_dir, exist_ok=True)
        

        x, torqueDown, torqueUp, mjc_model_file, geometry_vals, torque_curve = customize(
            input_params, 
            strength=params.get('strength', "default"), 
            model_dir=None, 
            output_dir=output_dir
        )
        

        results = {
            'torqueDown': float(torqueDown),
            'torqueUp': float(torqueUp),
            'mjcModelFile': mjc_model_file,
            'geometryValues': geometry_vals.tolist() if hasattr(geometry_vals, 'tolist') else geometry_vals,
            'torqueCurve': torque_curve.tolist() if hasattr(torque_curve, 'tolist') else torque_curve,
            'dimensions': {
                'd1': params.get('d1'),
                'd2': params.get('d2'),
                'd3': params.get('d3'),
                'w1': params.get('w1'),
                'w2': params.get('w2'),
                'w3': params.get('w3'),
                'l1': params.get('l1'),
                'l2': params.get('l2'),
                'l3': params.get('l3')
            }
        }
        
        results_file = os.path.join(current_dir, 'optimization_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        

        print(f"Results saved to {output_dir}")
        print(f"Model file: {mjc_model_file}")
        print(f"Torque down: {torqueDown:.2f}")
        print(f"Torque up: {torqueUp:.2f}")
        
    except Exception as e:
        print(f"Error running optimization: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()