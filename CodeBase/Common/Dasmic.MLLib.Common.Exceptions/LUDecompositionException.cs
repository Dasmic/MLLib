using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Common.Exceptions
{
    public class LUDecompositionException : Exception
    {
        public LUDecompositionException(
            string message,
            Exception innerException)
        {

        }

        public LUDecompositionException()
        {

        }

        public override string Message
        {
            get
            {
                return Resources.strings_messages.exception_lu_decomposition;
            }
        }
    }
}
