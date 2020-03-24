from data_processor.fcm_data import FcmDataProcessor

class FcmBci(FcmDataProcessor):
    def __init__(self):
        
        """ The FcmBci object initializes with a universe of discourse with a range [0,1].  """
        
        self.data = pd.DataFrame()
        self.universe = np.arange(0, 1.01, 0.01)
