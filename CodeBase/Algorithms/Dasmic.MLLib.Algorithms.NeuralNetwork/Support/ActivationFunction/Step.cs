using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction
{
    public class Step: IActivationFunction
    {
        public double GetValue(double weightValueProduct)
        {            
            double computedValue= weightValueProduct > 0 ? 1 : -1; //Convert to step function
            return computedValue;
        }

        public double GetDerivativeValue(double value)
        {
            return 0;
        }
    }
}
