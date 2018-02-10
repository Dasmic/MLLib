using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Algorithms.SupportVectorMachine
{
    public class ModelKernel
    {
        public List<double> Alphas { get; set; } //Store non-zero alpha
        public List<double> TargetValues { get; set; } //Store non-zero alpha
        public List<double[]> SupportVectors { get; set; } //values with non-zero alphas
        public double Threshold { get; set; }
        public IKernel Kernel;

        public ModelKernel()
        {
            Alphas = new List<double>();
            TargetValues = new List<double>();
            SupportVectors = new List<double[]>();
        }
    }
}
