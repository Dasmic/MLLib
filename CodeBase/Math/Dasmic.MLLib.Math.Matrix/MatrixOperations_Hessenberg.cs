using Dasmic.MLLib.Common.Exceptions;

namespace Dasmic.MLLib.Math.Matrix
{
    public partial class MatrixOperations
    {
        /// <summary>
        /// Return true is matrix is Hessenberg or not
        /// </summary>
        /// <param name="matrix"></param>
        /// <returns></returns>
        public bool IsHessenberg(double[][] matrix)
        {
            bool flag = true;
            for (int col=0;col < matrix.Length-2; col++)
            {
                for (int row = 2 + col; row < matrix.Length; row++)
                {
                    if (matrix[col][row] != 0)
                    {
                        flag = false;
                        break;
                    }
                }
            }
            return flag;
        }
        
        /// <summary>
        /// Returns the Hessenberg form of a square matrix
        /// </summary>
        /// <param name="matrix"></param>
        /// <returns></returns>
        public double[][] Hessenberg(double [][] matrix, ref double [][] U, bool shouldVerify=true)
        {
            if(shouldVerify)
            {
                GeneralVerifyAndIfSquare(matrix);
            }
            if (IsHessenberg(matrix))
                return matrix; //Return as is

            //Compute Hessenberg using Householder  Reduction
            //As per: http://www.aip.de/groups/soe/local/numres/bookcpdf/c11-5.pdf
            //Gaussian method is 2x faster than Householder reduction
            //However as per: http://www.ams.org/journals/mcom/1969-23-108/S0025-5718-1969-0258255-3/S0025-5718-1969-0258255-3.pdf
            //Gaussian elimination is unstable.
            //We need a stable method due to giving guarantees for any input

            //Create copy of main matrix
            double[][] H=null;
            CopyMatrix(matrix, ref H);

            for (int col=0; col<H.Length-2;col++)
            {
                double alpha = 0;
                for (int rowX = col+1; rowX < H.Length; rowX++)
                {
                    alpha += System.Math.Pow(H[col][rowX], 2);
                }
                alpha = System.Math.Sqrt(alpha);

                //Form matrix w
                //w as per http://web.csulb.edu/~tgao/math423/s93.pdf
                double[][] w = new double[1][]; //w = a - r
                double wMag = 0;//Magnitude of vector
                w[0] = new double[H[0].Length];
                for (int row=0;row < w[0].Length;row++)
                {
                    if(row <= col)
                        w[0][row] = 0;// Due to substraction of r - a
                    else if (row == col + 1)
                    {
                        w[0][row] = H[col][row] - alpha;
                    }
                    else
                        w[0][row] = H[col][row];
                    wMag += System.Math.Pow(w[0][row], 2);
                }
                //wMag = System.Math.Sqrt(wMag);
                //Divide each row by Mag
                //for (int row = 0; row < w[0].Length; row++)
                //    w[0][row] = w[0][row] / wMag;
                double[][] wwt = Multiply(w, Transpose(w));
                wwt = MultiplyByScalar(wwt, (2.0)/wMag);
                double[][] P = IdentityMatrix(matrix.Length);
                P = Substract(P, wwt);
                //Create new H
                //Multiply on both sides, if it it QR only multiply in left
                H = Multiply(P,H);// H,P); 
                H = Multiply(H,P);//P, H);
                if (col == 0)
                    U = P;
                else
                    U = Multiply(U,P);
            }
            return H;

        }
    }
} 
