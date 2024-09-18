class ComparisonMethodBase:
    def __init__(self, **params):
        self.params = params
    
    @staticmethod
    def get_name():
        """
        Returns the name of the comparison method.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @staticmethod
    def get_parameter_info(data):
        """
        Returns a list of dictionaries describing the parameters needed by the method.
        Each dictionary can include parameter name, default value, type, description, etc.
        """
        return []

    def compare(self, data, bdv_column):
        """
        Performs the comparison on the data.
        Returns a DataFrame with the results.
        """
        raise NotImplementedError("Subclasses should implement this method.")