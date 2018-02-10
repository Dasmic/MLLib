using Dasmic.MLLib.Common.MLCore;

namespace Dasmic.MLLib.Algorithms.NeuralNetwork
{
    public abstract class BuildBase : MLLib.Common.MLCore.BuildBase
    {
        protected double _alpha;
        protected double _noOfEpoch;
        protected double _weightBaseValue;
        protected double _threshold; //Value of derivative

        public BuildBase()
        {
            _alpha = .3;
            _noOfEpoch = 1000;
            _weightBaseValue = .005;
            _threshold = .01;
        }

        public override abstract Common.MLCore.ModelBase BuildModel(double[][] trainingData,
                           string[] attributeHeaders,
                           int indexTargetAttribute);

        public int GetNumberOfHiddenUnits(int noOfUnitsOutputLayer, double scalingFactor, 
                                                int noOfAttributes)
        {
            
                //Use equation from:
                //https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw/1097#1097
                double value = (double)_trainingData[0].Length
                                    / (double)((noOfUnitsOutputLayer +
                                    noOfAttributes) * scalingFactor);
                return (int)System.Math.Ceiling(value);
            
        }
    }
}
