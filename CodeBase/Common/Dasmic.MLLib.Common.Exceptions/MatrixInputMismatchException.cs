using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Common.Exceptions
{
    public class MatrixInputMismatchException:Exception
    {

        public MatrixInputMismatchException(string message,
            Exception innerException) : base(message, innerException)
            {

            }

        public MatrixInputMismatchException()
        {

        }

        public override string Message
        {
            get
            {
                return Resources.strings_messages.exception_matrix_input_mismatch;
            }
        }
    }
}
