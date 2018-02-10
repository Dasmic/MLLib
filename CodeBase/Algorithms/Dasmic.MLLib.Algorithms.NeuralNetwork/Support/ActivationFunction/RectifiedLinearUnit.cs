namespace Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction
{    
    public class RectifiedLinearUnit : IActivationFunction
    {
        public double GetValue(double weightValueProduct)
        {
            //Apply ReLu function
            double computedValue = weightValueProduct > 0 ? weightValueProduct : 0;
            return computedValue;
        }

        /// <summary>
        /// See derivation for explanation here:
        /// https://stackoverflow.com/questions/30236856/how-does-the-back-propagation-algorithm-deal-with-non-differentiable-activation
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public double GetDerivativeValue(double value)
        {
            return 1.0; //Can also be 0 in some cases
        }
    }
}
