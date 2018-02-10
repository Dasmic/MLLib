using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Math.Matrix
{
    public partial class MatrixOperations
    {
        public double[][] IdentityMatrix(int n)
        {
            double[][] mI = new double[n][];

            Parallel.For(0, n, idx =>
            {
                mI[idx] = new double[n];
                mI[idx][idx] = 1;
            });

            return mI;
        }
    }
}
