using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Algorithms.SupportVectorMachine
{
    /*
     * Simple Example: x = (x1, x2, x3); y = (y1, y2, y3). Then for the function f(x) = (x1x1, x1x2, x1x3, x2x1, x2x2, x2x3, x3x1, x3x2, x3x3), the kernel is K(x, y ) = (<x, y>)^2.

        Let's plug in some numbers to make this more intuitive: suppose x = (1, 2, 3); y = (4, 5, 6). Then:
        f(x) = (1, 2, 3, 2, 4, 6, 3, 6, 9)
        f(y) = (16, 20, 24, 20, 25, 30, 24, 30, 36)
        <f(x), f(y)> = 16 + 40 + 72 + 40 + 100+ 180 + 72 + 180 + 324 = 1024

        A lot of algebra. Mainly because f is a mapping from 3-dimensional to 9 dimensional space.

        Now let us use the kernel instead: 
        K(x, y) = (4 + 10 + 18 ) ^2 = 32^2 = 1024
        Same result, but this calculation is so much easier.
     */
    internal class KernelPolynomial:IKernel
    {
        private int _degree;

        public KernelPolynomial(int degree)
        {
            _degree = degree;
        }

        public double compute(double[] v1, double[] v2)
        {
            double value = 0;
            for (int ii = 0; ii < v1.Length; ii++)
            {
                value += Math.Pow(v1[ii] * v2[ii],_degree);
            }
            return value;
        }
    }
}
