using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Algorithms.SupportVectorMachine
{
    class KernelRadialBasisFunction : IKernel
    {
        double _gamma;

        public KernelRadialBasisFunction(double gamma)
        {
            _gamma = gamma;
        }

        public double compute(double[] v1, double[] v2)
        {
            double value=0;
            for (int ii = 0; ii < v1.Length; ii++)
            {
                value += Math.Pow((v1[ii] - v2[ii]), 2);
            }
            double power = Math.Exp(0 - (_gamma * value)); 
            return power;
        }
    }
}
