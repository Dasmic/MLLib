using Dasmic.Portable.Core;
using System.Threading.Tasks;
using System;

namespace Dasmic.MLLib.Algorithms.SupportVectorMachine
{
    /// <summary>
    /// Ref: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf
    /// </summary>
    public class SequentialMinimalOptimization
    {
        double[] _lagrangeMul;//Last Lagrange is b
        double[] _error_cache;//Last Lagrange is b
        double[][] _trainingData;
        int _indexTarget;
        double _C, _tolerance;
        long _maxIterations;
        double _epsilon; //Changed on 5/26 - //In mathematics (particularly calculus), an arbitrarily small positive quantity is commonly denoted as epsilon
        Random _random;
        IKernel _kernel;
        //double[] _coeffs;
        double _threshold; //Representing b
        double _bDelta;

        public SequentialMinimalOptimization()
        {
            _random = new Random();
            _maxIterations = 10000;
            _C = 2;
            _tolerance = .0001;
            _epsilon = .001;
            _threshold = 0;
        }


        public void setParameters(double C,
                            double tolerance, 
                            double eps,
                            long maxIterations)
        {
            _C = C;
            _tolerance = tolerance;
            _epsilon = eps;
            _maxIterations = maxIterations;
        }


        /// <summary>
        /// Get coefficients from the Langrange Multipliers
        /// 
        /// This function assumed the last attribute in trainingData is the target value
        /// </summary>
        /// <returns></returns>
        public ModelKernel computeSupportVectors(double[][] trainingData,
                                        int indexTargetAttribute,
                                        IKernel kernel,
                                        int maxThreads)
        {
            //Update private Vars -----------------
            _kernel = kernel;
            _lagrangeMul = new double[trainingData[0].Length];
            _error_cache = new double[trainingData[0].Length];           
            SupportFunctions.InitArray(_lagrangeMul, maxThreads);
            _indexTarget = indexTargetAttribute;
            _trainingData = trainingData; 
            //---------------------------

            //main routine:            
            int numChanged = 0;
            bool examineAll = true;
            while (numChanged > 0 || examineAll)
            {
                numChanged = 0;
                if (examineAll)
                {
                    //loop I over all training examples
                    for (int row = 0; row < trainingData[0].Length; row++)
                    {
                        numChanged += examineExample(row);
                    }
                }
                else
                {
                    //loop I over examples where alpha is not 0 & not C
                    for (int row = 0; row < trainingData[0].Length; row++)
                    {
                        if(_lagrangeMul[row] != 0 && _lagrangeMul[row] != _C)
                        {
                            double[] data = SupportFunctions.GetLinearArray(trainingData, row);
                            numChanged += examineExample(row);
                        }
                    }
                }
               
                if (examineAll)
                    examineAll = false;
                else if (numChanged == 0)
                    examineAll = true;
            }
            return getModelKernel();                      
        }

       

        /// <summary>
        /// 
        /// </summary>
        /// <param name="i2"></param>
        /// <returns></returns>
        private int examineExample(long row2)
        {
            //double[] i2 =
            //                SupportFunctions.getLinearArray(_trainingData, row2);
            double y2 = _trainingData[_indexTarget][row2];
            double alph2 = _lagrangeMul[row2];// Lagrange multiplier for i2
            double E2 = getError(row2);
            double r2 = E2 * y2;
            long row1;
            if ((r2 < -_tolerance && alph2 < _C) || (r2 > _tolerance && alph2 > 0))
            {
                //if (getNumberNonZeroandCAlphas() > 1)
                //{
                //if number of non-zero and non-C alphas > 0
                row1 = getRowForSecondChoiceHeuristic(E2);
                if(row1 >=0) //if row<- then number of Zero and Calpha < 0
                     if (takeStep(row1, row2))
                        return 1;
                 //}
                //loop over all non-zero and non-C alpha, starting at a random point
                int startIdx = _random.Next(0, _lagrangeMul.Length-1);
                for (long row=startIdx;row<_lagrangeMul.Length;row++)
                {
                    //smo paper is adding more randomization here
                    if (_lagrangeMul[row] != 0 && 
                                _lagrangeMul[row] != _C)
                    {
                       row1 = row; 
                       if (takeStep(row1, row2))
                            return 1;
                    }
                }
                //loop over all possible i1, starting at a random point
                //Will reach here if value still not returned
                for (long row = startIdx; row < _lagrangeMul.Length; row++)
                {
                        row1 = row;
                        if (takeStep(row1, row2))
                            return 1;
                }               
            }
            return 0;
    } //endprocedure


        //target = desired output vector
        //point = training point matrix
        //procedure takeStep(i1, i2)
        private bool takeStep(long row1, long row2)
        {
            if (row1 == row2) return false;
            double alph1 = _lagrangeMul[row1];// Lagrange multiplier for i1
            double alph2 = _lagrangeMul[row2];// Lagrange multiplier for i2
            double y1 = _trainingData[_indexTarget][row1];
            double y2 = _trainingData[_indexTarget][row2];
            //E1 = SVM output on point[i1] – y1(check in error cache)
            double E1 = getError(row1);
            double E2 = getError(row2);

            double s = y1 * _trainingData[_indexTarget][row2];
            //Compute L, H via equations(13) and(14)
            double L = getL(row1, row2);
            double H = getH(row1, row2);

            if (L == H)
                return false;

            //double [] point1 = SupportFunctions.getLinearArray(_trainingData, row1);
            //double[] point2 = SupportFunctions.getLinearArray(_trainingData, row2);
            double k11 = computeKernel(row1, row1);// _kernel.compute(point1, point1);
            double k12 = computeKernel(row1, row2); //_kernel.compute(point1, point1);
            double k22 = computeKernel(row2, row2); // _kernel.compute(point1, point2);
            double eta = k11 + k22 - 2 * k12;

            double a2 = 0;
            if (eta > 0)
            {
                a2 = alph2 + y2 * (E1 - E2) / eta;
                if (a2 < L)
                    a2 = L;
                else if (a2 > H)
                    a2 = H;
            }
            else
            {
                //Objective function computation taken from:
                //http://web.cs.iastate.edu/~honavar/smo-svm.pdf
                double c1 = eta / 2.0;
                double c2 = y2 * (E1 - E2) * -eta * alph2;
                //Lobj = objective function at a2 = L
                double Lobj = c1 * L * L + c2 * L;
                //Hobj = objective function at a2 = H
                double Hobj = c1 * H * H + c2 * H;

                if (Lobj < Hobj - _epsilon)
                    a2 = L;
                else if (Lobj > Hobj + _epsilon)
                    a2 = H;
                else
                    a2 = alph2;
            }
            if ( Math.Abs(a2 - alph2)  < 
                        _epsilon * (a2 + alph2 + _epsilon))
                return false;
            double a1 = alph1 + s * (alph2 - a2);

            //Update threshold to reflect change in Lagrange multipliers
            updateThreshold(alph1,a1,E1,y1,alph2,a2,E2,y2,k11,k12,k22);
            //Update weight vector to reflect change in a1 & a2, if SVM is linear
            //updateWeightVectorIfLinearSVM(); //treat all SVM same right now
            //Update error cache using new Lagrange multipliers
            updateErrorCache(row1,alph1, a1,  y1, row2,alph2, a2, y2);
            //Store alphas first as they will be needed by the other functions following it
            //Store a1 in the alpha array
            _lagrangeMul[row1] = a1;
            //Store a2 in the alpha array
            _lagrangeMul[row2] = a2;
            
            return true;
        } //End of procedure


        #region Support Functions


        /// <summary>
        /// Returns the ModelKernel
        /// </summary>
        /// <returns></returns>
        private ModelKernel getModelKernel()
        {
            ModelKernel mk = new ModelKernel();
            mk.Kernel = _kernel;
            for (long row = 0; row < _lagrangeMul.Length; row++)
            {
                if (_lagrangeMul[row] > 0)
                {                   
                    //Get Support Vectors
                    double[] sv =  //Do not get target value index
                        SupportFunctions.GetLinearArray(_trainingData, 
                                            row, _trainingData.Length - 2 );
                    //Get alpha
                    mk.Alphas.Add(_lagrangeMul[row]);
                    mk.TargetValues.Add(_trainingData[_indexTarget][row]);
                    mk.SupportVectors.Add(sv);
                    //getThreshold
                    
                }
            }
            mk.Threshold = _threshold;
            return mk;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="a1">New alpha1</param>
        /// <param name="a2">New alpha2</param>
        private void updateThreshold(double alph1, double a1,double E1, double y1, 
                                     double alph2, double a2,double E2, double y2,
                                     double k11,double k12,double k22)
        {
            double bnew = 0;
            double b = _threshold; //Last Index           
            if (a1 > 0 && a1 < _C)
                bnew = b + E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12;
            else
            {
                if (a2 > 0 && a2 < _C)
                    bnew = b + E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22;
                else
                {
                    double b1 = b + E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12;
                    double b2 = b + E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22;
                    bnew = (b1 + b2) / 2;
                }
            }
            _bDelta = bnew - b;
            _threshold = bnew;            
        }


        /// <summary>
        /// Update the weight vector if Linear SVM
        /// This will save time in evaluation of learned function
        /// For non-linear SVM use of Kernel will have computation overheads
        /// </summary>
        private void updateWeightVectorIfLinearSVM()
        {
            for(int row=0;row<_lagrangeMul.Length;row++)
            {
                if(_lagrangeMul[row] > 0)
                {

                }
            }
        }

        /// <summary>
        /// Ref: http://web.cs.iastate.edu/~honavar/smo-svm.pdf
        /// </summary>
        /// <param name="alph1"></param>
        /// <param name="a1"></param>
        /// <param name="y1"></param>
        /// <param name="alph2"></param>
        /// <param name="a2"></param>
        /// <param name="y2"></param>
        private void updateErrorCache(long row1,double alph1, double a1, double y1, 
                                      long row2,double alph2, double a2, double y2)
        {
            double t1 = y1 * (a1 - alph1);
            double t2 = y2 * (a2 - alph2);

            for (int row = 0; row < _lagrangeMul.Length; row++)
            {
                //double[] pointi = SupportFunctions.getLinearArray(_trainingData, row);
                //double[] pointi1 = SupportFunctions.getLinearArray(_trainingData, row1);
                //double[] pointi2 = SupportFunctions.getLinearArray(_trainingData, row2);
                _error_cache[row] += t1 * computeKernel(row1, row) +
                                        t2 * computeKernel(row2, row) - _bDelta;

            }
            _error_cache[row1] = 0;
            _error_cache[row2] = 0;
        }
        
        
        private double getError(long row)
        {
            double E;
            double alpha = _lagrangeMul[row];

            if (alpha > 0 && alpha < _C)
                E = _error_cache[row];
            else
                E = getLearnedFunctionValue(row) -
                        _trainingData[_indexTarget][row];
            return E;
        }

        /// <summary>
        /// Return the learned function value
        /// </summary>
        /// <param name="row"></param>
        /// <returns></returns>
        private double getLearnedFunctionValue(long row)
        {
            double value=0;

            for (int idx = 0; idx < _lagrangeMul.Length; idx++)
            {
                if (_lagrangeMul[idx] > 0)
                    value += _lagrangeMul[idx] * _trainingData[_indexTarget][idx]
                                        * computeKernel(idx, row);
                
            }
            value -= _threshold; //Threshold is always substracted
            return value;
        }

        /// <summary>
        /// Returns the index of row1 based on Heuristic
        /// 
        /// Ref: http://web.cs.iastate.edu/~honavar/smo-svm.pdf
        /// 
        /// </summary>
        /// <returns></returns>
        private long getRowForSecondChoiceHeuristic(double E2)
        {
            long k, idx;
            double tmax;
            double E1, tmp;
            idx = -1;
            tmax = 0;
            for(k=0;k<_lagrangeMul.Length;k++)
            {
                if(_lagrangeMul[k] > 0 &&
                    _lagrangeMul[k] < _C)
                {
                    E1 = _error_cache[k];
                    tmp = Math.Abs(E1 - E2);
                    if(tmp>tmax)
                    {
                        tmax = tmp;
                        idx = k;
                    }
                }
            }
            return idx;
        }

        private double getL(long row1, long row2)
        {
            double L;
            double alpha1 = _lagrangeMul[row1];
            double alpha2 = _lagrangeMul[row2];
            if (_trainingData[_indexTarget][row1]
                ==_trainingData[_indexTarget][row2])
            {
                L = alpha2 - alpha1 > 0 ? alpha2 - alpha1 : 0;
            }
            else
                L = alpha2 + alpha1 - _C > 0 ? alpha2 + alpha1 - _C : 0;
            return L;
        }

        private double getH(long row1, long row2)
        {
            double H;
            double alpha1 = _lagrangeMul[row1];
            double alpha2 = _lagrangeMul[row2];
            if (_trainingData[_indexTarget][row1]
                == _trainingData[_indexTarget][row2])
            {
                H = _C + alpha2 - alpha1 > _C ? _C + alpha2 - alpha1 : _C;
            }
            else
                H = alpha2 + alpha1 > _C ? alpha2 + alpha1  : _C;
            return H;
        }

        private double computeKernel(long row1,long row2)
        {
            double value = 0;
            double[] point1 = SupportFunctions.GetLinearArray(_trainingData, row1, _trainingData.Length-2);
            double[] point2 = SupportFunctions.GetLinearArray(_trainingData, row2, _trainingData.Length - 2);

            value = _kernel.compute(point1, point2);
            return value;
        }

       

        #endregion


    }
}
