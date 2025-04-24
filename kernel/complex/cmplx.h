
#ifndef CMPLX_H
#define CMPLX_H
typedef float2 cfloat;

inline cfloat cadd(cfloat a, cfloat b)
{
    return (cfloat)(a.x + b.x,
                    a.y + b.y);
}

inline cfloat csub(cfloat a, cfloat b)
{
    return (cfloat)(a.x - b.x,
                    a.y - b.y);
}

inline cfloat cmul(cfloat a, cfloat b)
{
    return (cfloat)(a.x * b.x - a.y * b.y,
                    a.x * b.y + a.y * b.x);
}

inline float cnorm2(cfloat a)
{
    return a.x * a.x + a.y * a.y;
}

#endif // CMPLX_H
