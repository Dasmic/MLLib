using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Algorithms.SupportVectorMachine
{
    internal class KernelLinear:IKernel
    {
        public double compute(double[] v1, double[] v2)
        {
            double value=0;
            for(int ii=0;ii<v1.Length;ii++)
            {
                value += v1[ii] * v2[ii];
            }
            return value;
        }
    }
}
