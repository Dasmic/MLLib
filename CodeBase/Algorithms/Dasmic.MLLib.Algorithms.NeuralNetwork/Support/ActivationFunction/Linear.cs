
namespace Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction
{
    public class Linear:IActivationFunction
    {
        public double GetValue(double weightValueProduct)
        {
            return weightValueProduct;
        }

        public double GetDerivativeValue(double value)
        {
            return 1;
        }
    }
}
