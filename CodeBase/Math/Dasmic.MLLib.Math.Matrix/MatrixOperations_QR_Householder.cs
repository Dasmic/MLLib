using System;
using System.Threading.Tasks;
using Dasmic.MLLib.Common.Exceptions;

namespace Dasmic.MLLib.Math.Matrix
{
    public partial class MatrixOperations
    {                      
        /// <summary>
        /// Computed QR of a given Hessenberg Matrix using Householder rotation
        /// Will not work for complex numbers
        /// 
        /// CAUTION: Matrix passed should be Hessenberg Matrix
        /// </summary>
        /// <param name="matrix">Hessenberg Matrix</param>
        /// <param name="Q"></param>
        /// <param name="R"></param>
        /// <returns>Matrix in QR Form</returns>
        public void QRDecomposition_Householder(
                                double[][]matrix,                                
                                ref double[][] Q,
                                ref double[][] R,                                
                                bool shouldVerify=true)
        {
            if (shouldVerify)
            {
                GeneralVerifyAndIfSquare(matrix);
            }
           

            //Using method of computing Hessenberg using Householder  Reduction
            //As per: http://www.aip.de/groups/soe/local/numres/bookcpdf/c11-5.pdf
            //Gaussian method is 2x faster than Householder reduction
            

            //Create copy of main matrix
            double[][] H = null;
            CopyMatrix(matrix, ref H);
            //Use method in 
            for (int col = 0; col < H.Length - 1; col++)
            {
                double alpha = 0;
                for (int rowX = col; rowX < H.Length; rowX++)
                {
                    alpha += System.Math.Pow(H[col][rowX], 2);
                }
                alpha = System.Math.Sqrt(alpha);

                //Form matrix u
                //w=u as per https://math.la.asu.edu/~gardner/QR.pdf
                double[][] w = new double[1][];
                double wMag = 0;//Magnitude of vector
                w[0] = new double[H[0].Length];
                for (int row = 0; row < w[0].Length; row++)
                {
                    if (row < col)
                        w[0][row] = 0;// Due to substraction of r - a
                    else if (row == col)
                    {
                        w[0][row] = H[col][row] - alpha;
                    }
                    else
                        w[0][row] = H[col][row];
                    wMag += System.Math.Pow(w[0][row], 2);
                }
                
                double[][] wwt = Multiply(w, Transpose(w));
                wwt = MultiplyByScalar(wwt, (2.0) / wMag);
                double[][] P = IdentityMatrix(matrix.Length);
                P = Substract( P, wwt);
                //Create new H
                H = Multiply(P, H);
                //H = Multiply(H, P);//P, H);
                if (col == 0)
                    Q = P;
                else
                    Q = Multiply(Q, P);
            }
            R = H;            
        } //QRDecomposition_Householder
    }
}
