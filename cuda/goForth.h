#ifndef _GOFORTH_H
#define _GOFORTH_H

#ifdef __cplusplus
extern "C"
#endif
void goForthAndMultiply(square_t *gridA, square_t *gridB, const long rows, 
                        const long cols, const long numSteps);

#endif //_GOFORTH_H