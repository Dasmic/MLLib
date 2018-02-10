using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Common.Exceptions
{
        public class InvalidMatrixException : Exception
        {
            public InvalidMatrixException(
                string message,
                Exception innerException) : base(message, innerException)
            {

            }

            public InvalidMatrixException()
            {

            }

            public override string Message
            {
                get
                {
                    return Resources.strings_messages.exception_invalid_matrix;
                }
            }
        }
}
