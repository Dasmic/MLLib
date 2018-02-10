
namespace Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction
{
    public class HyperbolicTangent : IActivationFunction
    {
        public double GetValue(double weightValueProduct)
        {
            //Apply SigMoid function            
            return System.Math.Tanh(weightValueProduct);// computedValue;
        }

        public double GetDerivativeValue(double value)
        {
            double tVal = System.Math.Tanh(value);
            return 1.0 - (tVal * tVal);
        }
    }
}
