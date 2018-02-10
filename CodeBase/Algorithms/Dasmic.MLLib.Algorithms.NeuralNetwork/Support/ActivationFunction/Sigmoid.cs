
namespace Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction
{
    public class Sigmoid : IActivationFunction
    {
        public double GetValue(double weightValueProduct)
        {
            //Apply SigMoid function
            double computedValue = 1.0 / (1.0 + System.Math.Exp(0 -
                                        weightValueProduct));
            return computedValue;
        }

        public double GetDerivativeValue(double value)
        {
            return value * (1.0 - value);
        }
    }
}
