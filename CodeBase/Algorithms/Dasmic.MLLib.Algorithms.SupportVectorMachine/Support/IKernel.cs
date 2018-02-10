using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Algorithms.SupportVectorMachine
{
    public interface IKernel
    {
        double compute(double[] v1, double[] v2);
    }
}
