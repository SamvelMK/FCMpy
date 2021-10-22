
if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from fcmpy.ml import AHL
    from fcmpy.ml import HebbianValidate
    from simulator.simulator import FcmSimulator
    
    w_init = np.asarray([[0,-0.4,-0.25,0,0.3],[0.36,0,0,0,0],
                        [0.45,0,0,0,0],[-0.9,0,0,0,0],
                        [0,0.6,0,0.3,0]])

    w_init_df = pd.DataFrame(w_init, columns=['C1', 'C2', 'C3', 'C4', 'C5'],
                            index = ['C1', 'C2', 'C3', 'C4', 'C5'])

    init_state = np.asarray([0.4,0.707,0.607,0.72,0.3])
    init_state_d = {'C1' : 0.40, 'C2': 0.7077, 'C3': 0.612, 'C4': 0.717, 'C5': 0.30}
    activation_pattern = {0:['C1'], 1:['C2', 'C3'], 2: ['C5'], 3: ['C4']}

    doc_values = {'C1':[0.68,0.74], 'C5':[0.74,0.8]}
    
    ahl = AHL(state_vector=init_state_d, weight_matrix=w_init_df, activation_pattern=activation_pattern,
                doc_values=doc_values)

    res = ahl.run(decay=0.03, learning_rate = 0.01, l=1, iterations=100,
                    transfer= 'sigmoid', thresh = 0.002, auto_learn=False,
                    b1=0.003, lbd1=0.1, b2=0.005, lbd2=1)

    print(res)

    validation = HebbianValidate(FcmSimulator=FcmSimulator)
    validation.validate(n_validations=100, doc_values=doc_values, concepts = list(init_state_d.keys()), weight_matrix=res)
    print(f"mean C1: { np.mean(validation.results['C1'])}")
    print(f"min C1: { np.min(validation.results['C1'])}")
    print(f"max C1: { np.max(validation.results['C1'])}")