// FastMath.h defines SIMD vector, matrix, and quaternion libraries
// NOTE: This library uses SSE4.1 extensions, which is not supported
// by AMD.
#ifndef _FASTMATH_H_
#define _FASTMATH_H_

#include <xmmintrin.h>
#include <smmintrin.h>
#include <cmath>

// Define D3D Matrix here so we don't have to replicate
#ifndef D3DMATRIX_DEFINED
typedef struct _D3DMATRIX {
	union {
		struct {
			float        _11, _12, _13, _14;
			float        _21, _22, _23, _24;
			float        _31, _32, _33, _34;
			float        _41, _42, _43, _44;

		};
		float m[4][4];
	};
} D3DMATRIX;
#define D3DMATRIX_DEFINED
#endif

#define FASTMATH_TRANSPOSE(row0, row1, row2, row3) {					\
	__m128 tmp3, tmp2, tmp1, tmp0;										\
																		\
	tmp0   = _mm_shuffle_ps((row0), (row1), _MM_SHUFFLE(1, 0, 1, 0));	\
	tmp2   = _mm_shuffle_ps((row0), (row1), _MM_SHUFFLE(3, 2, 3, 2));	\
	tmp1   = _mm_shuffle_ps((row2), (row3), _MM_SHUFFLE(1, 0, 1, 0));	\
	tmp3   = _mm_shuffle_ps((row2), (row3), _MM_SHUFFLE(3, 2, 3, 2));	\
																		\
	(row0) = _mm_shuffle_ps(tmp3, tmp2, _MM_SHUFFLE(1, 3, 1, 3));		\
	(row1) = _mm_shuffle_ps(tmp3, tmp2, _MM_SHUFFLE(0, 2, 0, 2));		\
	(row2) = _mm_shuffle_ps(tmp1, tmp0, _MM_SHUFFLE(1, 3, 1, 3));		\
	(row3) = _mm_shuffle_ps(tmp1, tmp0, _MM_SHUFFLE(0, 2, 0, 2));		\
	}

namespace Bits
{

#ifndef _MATH_PI_CONSTANTS_
#define _MATH_PI_CONSTANTS_
const float TwoPi = 3.1415926535f * 2.0f;
const float Pi = 3.1415926535f;
const float PiOver2 = 3.1415926535f / 2.0f;
const float PiOver4 = 3.1415926535f / 4.0f;
#endif

// 4x4 Matrix class using SIMD
__declspec(align(16)) class FastMatrix4
{
private:
	__m128 _rows[4];
public:
	friend class FastVector3;
	friend class FastQuaternion;

	// Default constructor does nothing
	__forceinline FastMatrix4() {}

	// Constructs the matrix based on the passed-in floating point array[rows][column]
	__forceinline FastMatrix4(float mat[4][4])
	{
		_rows[0] = _mm_set_ps(mat[0][0], mat[0][1], mat[0][2], mat[0][3]);
		_rows[1] = _mm_set_ps(mat[1][0], mat[1][1], mat[1][2], mat[1][3]);
		_rows[2] = _mm_set_ps(mat[2][0], mat[2][1], mat[2][2], mat[2][3]);
		_rows[3] = _mm_set_ps(mat[3][0], mat[3][1], mat[3][2], mat[3][3]);
	}

	// Copy constructor
	__forceinline FastMatrix4(const FastMatrix4& rhs)
	{
		_rows[0] = rhs._rows[0];
		_rows[1] = rhs._rows[1];
		_rows[2] = rhs._rows[2];
		_rows[3] = rhs._rows[3];
	}

	// Assignment operator
	__forceinline FastMatrix4& operator=(const FastMatrix4& rhs)
	{
		_rows[0] = rhs._rows[0];
		_rows[1] = rhs._rows[1];
		_rows[2] = rhs._rows[2];
		_rows[3] = rhs._rows[3];
		return *this;
	}

	// Returns this FastMatrix4 as a D3DMATRIX*
	// This function is kinda expensive because it has to return a row-major matrix
	__forceinline D3DMATRIX* ToD3D()
	{
		static union
		{
			__m128 _rows_d3d[4];
			D3DMATRIX _d3dm;
		};
		_rows_d3d[0] = _mm_shuffle_ps(_rows[0], _rows[0], _MM_SHUFFLE(0, 1, 2, 3));
		_rows_d3d[1] = _mm_shuffle_ps(_rows[1], _rows[1], _MM_SHUFFLE(0, 1, 2, 3));
		_rows_d3d[2] = _mm_shuffle_ps(_rows[2], _rows[2], _MM_SHUFFLE(0, 1, 2, 3));
		_rows_d3d[3] = _mm_shuffle_ps(_rows[3], _rows[3], _MM_SHUFFLE(0, 1, 2, 3));

		_MM_TRANSPOSE4_PS(_rows_d3d[0], _rows_d3d[1], _rows_d3d[2], _rows_d3d[3]);

		return &_d3dm;
	}

	// Multiplies this matrix by the rhs matrix, and stores the result in this matrix.
	void Multiply(const FastMatrix4& rhs)
	{
		__m128 rhs_row0 = rhs._rows[0];
		__m128 rhs_row1 = rhs._rows[1];
		__m128 rhs_row2 = rhs._rows[2];
		__m128 rhs_row3 = rhs._rows[3];

		// transpose the rhs matrix
		FASTMATH_TRANSPOSE(rhs_row0, rhs_row1, rhs_row2, rhs_row3);
		

		__m128 row0;
		__m128 c_x = _mm_dp_ps(_rows[0],rhs_row0,0xFF);
		row0.m128_f32[3] = c_x.m128_f32[0];
		__m128 c_y = _mm_dp_ps(_rows[0],rhs_row1,0xFF);
		row0.m128_f32[2] = c_x.m128_f32[0];
		__m128 c_z = _mm_dp_ps(_rows[0],rhs_row2,0xFF);
		row0.m128_f32[1] = c_x.m128_f32[0];
		__m128 c_w = _mm_dp_ps(_rows[0],rhs_row3,0xFF);

		row0.m128_f32[3] = c_x.m128_f32[0];
		row0.m128_f32[2] = c_y.m128_f32[0];
		row0.m128_f32[1] = c_z.m128_f32[0];
		row0.m128_f32[0] = c_w.m128_f32[0];

		__m128 row1;
		c_x = _mm_dp_ps(_rows[1],rhs_row0,0xFF);
		row1.m128_f32[3] = c_x.m128_f32[0];
		c_y = _mm_dp_ps(_rows[1],rhs_row1,0xFF);
		row1.m128_f32[2] = c_x.m128_f32[0];
		c_z = _mm_dp_ps(_rows[1],rhs_row2,0xFF);
		row1.m128_f32[1] = c_x.m128_f32[0];
		c_w = _mm_dp_ps(_rows[1],rhs_row3,0xFF);

		row1.m128_f32[3] = c_x.m128_f32[0];
		row1.m128_f32[2] = c_y.m128_f32[0];
		row1.m128_f32[1] = c_z.m128_f32[0];
		row1.m128_f32[0] = c_w.m128_f32[0];

		__m128 row2;
		c_x = _mm_dp_ps(_rows[2],rhs_row0,0xFF);
		row2.m128_f32[3] = c_x.m128_f32[0];
		c_y = _mm_dp_ps(_rows[2],rhs_row1,0xFF);
		row2.m128_f32[2] = c_x.m128_f32[0];
		c_z = _mm_dp_ps(_rows[2],rhs_row2,0xFF);
		row2.m128_f32[1] = c_x.m128_f32[0];
		c_w = _mm_dp_ps(_rows[2],rhs_row3,0xFF);
		
		row2.m128_f32[3] = c_x.m128_f32[0];
		row2.m128_f32[2] = c_y.m128_f32[0];
		row2.m128_f32[1] = c_z.m128_f32[0];
		row2.m128_f32[0] = c_w.m128_f32[0];

		__m128 row3;
		c_x = _mm_dp_ps(_rows[3],rhs_row0,0xFF);
		row3.m128_f32[3] = c_x.m128_f32[0];
		c_y = _mm_dp_ps(_rows[3],rhs_row1,0xFF);
		row3.m128_f32[2] = c_x.m128_f32[0];
		c_z = _mm_dp_ps(_rows[3],rhs_row2,0xFF);
		row3.m128_f32[1] = c_x.m128_f32[0];
		c_w = _mm_dp_ps(_rows[3],rhs_row3,0xFF);
		
		row3.m128_f32[3] = c_x.m128_f32[0];
		row3.m128_f32[2] = c_y.m128_f32[0];
		row3.m128_f32[1] = c_z.m128_f32[0];
		row3.m128_f32[0] = c_w.m128_f32[0];


		_rows[0] = row0;
		_rows[1] = row1;
		_rows[2] = row2;
		_rows[3] = row3;
	}

	// Adds the rhs matrix to this one, storing in this
	__forceinline void Add(FastMatrix4& rhs)
	{
		_rows[0] = _mm_add_ps(_rows[0], rhs._rows[0]);
		_rows[1] = _mm_add_ps(_rows[1], rhs._rows[1]);
		_rows[2] = _mm_add_ps(_rows[2], rhs._rows[2]);
		_rows[3] = _mm_add_ps(_rows[3], rhs._rows[3]);
	}

	// Subtracts this - rhs, storing in this
	__forceinline void Sub(FastMatrix4& rhs)
	{
		_rows[0] = _mm_sub_ps(_rows[0], rhs._rows[0]);
		_rows[1] = _mm_sub_ps(_rows[1], rhs._rows[1]);
		_rows[2] = _mm_sub_ps(_rows[2], rhs._rows[2]);
		_rows[3] = _mm_sub_ps(_rows[3], rhs._rows[3]);
	}

	// Given the passed in scale, constructs a Scale matrix
	__forceinline void CreateScale(float scale)
	{
		// scale 0 0 0
		_rows[0] = _mm_set_ss(scale);
		_rows[0] = _mm_shuffle_ps(_rows[0], _rows[0], _MM_SHUFFLE(0, 1, 1, 1));

		// 0 scale 0 0
		_rows[1] = _mm_set_ss(scale);
		_rows[1] = _mm_shuffle_ps(_rows[1], _rows[1], _MM_SHUFFLE(1, 0, 1, 1));

		// 0 0 scale 0
		_rows[2] = _mm_set_ss(scale);
		_rows[2] = _mm_shuffle_ps(_rows[2], _rows[2], _MM_SHUFFLE(1, 1, 0, 1));

		// 0 0 0 1
		_rows[3] = _mm_set_ss(1.0f);
	}

	// Given the angle (in radians), constructs a Rotation about the X axis
	__forceinline void CreateRotationX(float angle)
	{
		// 1 0 0 0
		_rows[0] = _mm_set_ss(1.0f);
		_rows[0] = _mm_shuffle_ps(_rows[0], _rows[0], _MM_SHUFFLE(0, 1, 1, 1));

		float cos_theta = cosf(angle);
		float sin_theta = sinf(angle);

		// 0 cos -sin 0
		_rows[1] = _mm_set_ps(0.0f, cos_theta, sin_theta * -1.0f, 0.0f);

		// 0 sin cos 0
		_rows[2] = _mm_set_ps(0.0f, sin_theta, cos_theta, 0.0f);
		
		// 0 0 0 1
		_rows[3] = _mm_set_ss(1.0f);
	}

	// Given the angle (in radians), constructs a Rotation about the Y axis
	__forceinline void CreateRotationY(float angle)
	{
		float cos_theta = cosf(angle);
		float sin_theta = sinf(angle);

		// cos 0 sin 0
		_rows[0] = _mm_set_ps(cos_theta, 0.0f, sin_theta, 0.0f);

		// 0 1 0 0
		_rows[1] = _mm_set_ss(1.0f);
		_rows[1] = _mm_shuffle_ps(_rows[1], _rows[1], _MM_SHUFFLE(1, 0, 1, 1));

		// -sin 0 cos 0
		_rows[2] = _mm_set_ps(sin_theta * -1.0f, 0.0f, cos_theta, 0.0f);

		// 0 0 0 1
		_rows[3] = _mm_set_ss(1.0f);
	}

	// Given the angle (in radians), constructs a Rotation about the Z axis
	__forceinline void CreateRotationZ(float angle)
	{
		float cos_theta = cosf(angle);
		float sin_theta = sinf(angle);

		// cos -sin 0 0
		_rows[0] = _mm_set_ps(cos_theta, sin_theta * -1.0f, 0.0f, 0.0f);

		// sin cos 0 0
		_rows[1] = _mm_set_ps(sin_theta, cos_theta, 0.0f, 0.0f);

		// 0 0 1 0
		_rows[2] = _mm_set_ss(1.0f);
		_rows[2] = _mm_shuffle_ps(_rows[2], _rows[2], _MM_SHUFFLE(1, 1, 0, 1));

		// 0 0 0 1
		_rows[3] = _mm_set_ss(1.0f);
	}

	// Has to be defined in fastmath.cpp because of circular dependency
	
	// Given the translation vector, constructs a translation matrix.
	void CreateTranslation(const FastVector3& translation);

	// Given the quaternion, constructs a rotation matrix
	void CreateFromQuaternion(const FastQuaternion& q);

	// Constructs a Look-At matrix
	// vUp MUST be normalized or bad things will happen
	void CreateLookAt(const FastVector3& vEye, const FastVector3& vAt, const FastVector3& vUp);

	// Constructs a Perspective FOV matrix
	void CreatePerspectiveFOV(float fFOVy, float fAspectRatio, float fNear, float fFar);

	// Identity matrix
	static const FastMatrix4 Identity;
};

// 3D vector class using SIMD
class FastVector3
{
private:
	__m128 _data;
	// _data.m128_f32[3] = x
	// _data.m128_f32[2] = y
	// _data.m128_f32[1] = z
	// _data.m128_f32[0] = w
public:

	// Default constructor does nothing
	__forceinline FastVector3() {}

	// Constructs a vector given the passed x, y, and z components.
	// w is set to 1.0f
	__forceinline FastVector3(float x, float y, float z)
	{
		_data = _mm_set_ps(x, y, z, 1.0f);
	}

	// Constructs a FastVector3 given an __m128
	__forceinline FastVector3(__m128 value)
	{
		_data = value;
	}

	// Copy constructor
	__forceinline FastVector3(const FastVector3& rhs)
	{
		_data = rhs._data;
	}

	// Assignment operator
	__forceinline FastVector3& operator=(const FastVector3& rhs)
	{
		_data = rhs._data;
		return *this;
	}

	// Returns the X component (index 3)
	__forceinline float GetX() const
	{
		return _data.m128_f32[3];
	}
	
	// Returns the Y component (index 2)
	__forceinline float GetY() const
	{
		return _data.m128_f32[2];
	}
	
	// Returns the Z component (index 1)
	__forceinline float GetZ() const
	{
		return _data.m128_f32[1];
	}

	// Returns the W component (index 0)
	__forceinline float GetW() const
	{
		return _data.m128_f32[0];
	}
	
	// Sets the x, y, and z components to passed values.
	// w is set to 1.0f
	__forceinline void Set(float x, float y, float z)
	{
		_data = _mm_set_ps(x, y, z, 1.0f);
	}

	// Sets the X component
	__forceinline void SetX(float x)
	{
		// set index 0 of temp to x
		__m128 temp = _mm_set_ss(x);
		// insert the value into the X component
		_data = _mm_insert_ps(_data, temp, 0x30);
	}
	
	// Sets the Y component
	__forceinline void SetY(float y)
	{
		// set index 0 of temp to y
		__m128 temp = _mm_set_ss(y);
		// insert the value into the Y component
		_data = _mm_insert_ps(_data, temp, 0x20);
	}
	
	// Sets the Z component
	__forceinline void SetZ(float z)
	{
		// set index 0 of temp to z
		__m128 temp = _mm_set_ss(z);
		// insert the value into the Z component
		_data = _mm_insert_ps(_data, temp, 0x10);
	}

	// Computes the dot product between this vector and rhs.
	// Returns the float result.
	__forceinline float Dot(const FastVector3& rhs) const
	{
		__m128 temp = _mm_dp_ps(_data,rhs._data,0xE1);
		return temp.m128_f32[0]; // fixme -okay
	}

	// Adds this vector to rhs, storing in this
	__forceinline void Add(const FastVector3& rhs)
	{
		_data = _mm_add_ps(_data,rhs._data);
	}

	// Subtracts this - rhs, storing in this
	__forceinline void Sub(const FastVector3& rhs)
	{
		_data = _mm_sub_ps(_data,rhs._data);
	}

	// Does a scalar multiply by scalar
	__forceinline void Multiply(float scalar)
	{
		__m128 temp = _mm_set_ps1(scalar);
		_data = _mm_mul_ps(_data, temp);
	}

	// Normalizes this vector
	__forceinline void Normalize()
	{
		//Calculate length squared (data dot data)
		// The mask 0xEE will dot the x, y, and z components,
		// and store it in result's x, y, and z.
		__m128 temp = _mm_dp_ps(_data,_data,0xEE);
		
		//this wil store (1 / length) in x, y, and z
		temp = _mm_rsqrt_ps(temp);

		//multiply all components by (1 / length)
		_data = _mm_mul_ps(_data,temp);
	}

	// Returns the length squared of this vector
	__forceinline float LengthSquared() const
	{
		__m128 temp = _mm_dp_ps(_data,_data,0xE1);
		return temp.m128_f32[0]; // fixme - okay
	}

	// Returns the length of this vector
	__forceinline float Length() const
	{
		__m128 temp = _mm_dp_ps(_data,_data,0xEE);
		temp = _mm_sqrt_ps(temp);
		return temp.m128_f32[3]; // fixme - okay
	}

	// Does a cross product between lhs and rhs, returning the result vector by value
	__forceinline friend FastVector3 Cross(const FastVector3& lhs, const FastVector3& rhs)
	{
		//shuffle A to <ay,az,ax>
		//shuffle B to <bz,bx,by>
		__m128 tempA = _mm_shuffle_ps(lhs._data,lhs._data,_MM_SHUFFLE(2,1,3,0));
		__m128 tempB = _mm_shuffle_ps(rhs._data,rhs._data,_MM_SHUFFLE(1,3,2,0));
		
		__m128 templhs = _mm_mul_ps(tempA,tempB);

		//shuffle A to <az,ax,ay>
		//shuffle B to <by,bz,bx>
		tempA = _mm_shuffle_ps(lhs._data,lhs._data,_MM_SHUFFLE(1,3,2,0));
		tempB = _mm_shuffle_ps(rhs._data,rhs._data,_MM_SHUFFLE(2,1,3,0));

		__m128 temprhs = _mm_mul_ps(tempA,tempB);

		return FastVector3(_mm_sub_ps(templhs,temprhs)); // fixme - okay
	}

	// Interpolates between a and b, returning the result vector by value
	// result = a * (1.0f - f) + b * f
	__forceinline friend FastVector3 Lerp(const FastVector3 &a, const FastVector3 &b, float f)
	{
		
		__m128 tempA = _mm_mul_ps(a._data, _mm_set_ps1(1.0f - f));
		__m128 tempB = _mm_mul_ps(b._data, _mm_set_ps1(f));

		return FastVector3(_mm_add_ps(tempA,tempB)); // fixme -okay
	}

	// does a 4-way blend
	// result = a * fa + b * fb + c * fc + d * fd
	// fd = 1.0f - fa - fb - fc
	__forceinline friend FastVector3 Blend(const FastVector3& a, const FastVector3& b, const FastVector3& c, const FastVector3& d,
											float fa, float fb, float fc)
	{
		__m128 tempA = _mm_mul_ps(a._data,_mm_set_ps1(fa));
		__m128 tempB = _mm_mul_ps(b._data,_mm_set_ps1(fb));
		__m128 tempC = _mm_mul_ps(c._data,_mm_set_ps1(fc));

		float fd = 1.0f-fa-fb-fc;
		__m128 tempD = _mm_mul_ps(d._data,_mm_set_ps1(fd));

		tempA = _mm_add_ps(tempA,tempB);
		tempB = _mm_add_ps(tempC,tempD);

		return FastVector3(_mm_add_ps(tempA,tempB)); 
	}

	// Transforms this vector by the passed 4x4 matrix
	// w is set to 1.0f before the transform is done
	__forceinline void Transform(const FastMatrix4 &mat)
	{
		_data.m128_f32[0] = 1.0f;

		__m128 tempX = _mm_dp_ps(mat._rows[0],_data,0xFF);
		_data.m128_f32[3] = tempX.m128_f32[0];

		__m128 tempY = _mm_dp_ps(mat._rows[1],_data,0xFF);
		_data.m128_f32[2] = tempY.m128_f32[0];

		__m128 tempZ = _mm_dp_ps(mat._rows[2],_data,0xFF);
		_data.m128_f32[1] = tempZ.m128_f32[0];

	}

	// Rotates this vector by the passed quaternion
	void Rotate(const FastQuaternion& q);

	// Transforms this vector by the passed 4x4 matrix
	// w is set to 0.0f before the transform is done
	__forceinline void TransformAsVector(const FastMatrix4 &mat)
	{
		_data.m128_f32[0] = 0.0f;

		__m128 tempX = _mm_dp_ps(mat._rows[0],_data,0xFF);
		_data.m128_f32[3] = tempX.m128_f32[0];

		__m128 tempY = _mm_dp_ps(mat._rows[1],_data,0xFF);
		_data.m128_f32[2] = tempY.m128_f32[0];

		__m128 tempZ = _mm_dp_ps(mat._rows[2],_data,0xFF);
		_data.m128_f32[1] = tempZ.m128_f32[0];
	}

	friend class FastMatrix4;
	friend class FastQuaternion;

	static const FastVector3 Zero;
	static const FastVector3 UnitX;
	static const FastVector3 UnitY;
	static const FastVector3 UnitZ;
	static const FastVector3 NegativeUnitX;
	static const FastVector3 NegativeUnitY;
	static const FastVector3 NegativeUnitZ;
};

// Unit quaternion class using SIMD
class FastQuaternion
{
private:
	__m128 _data;
	// 3-1 is vector component
	// 0 is scalar component
public:
	// Default constructor does nothing
	FastQuaternion() {}

	// Constructs the quaternion given the normalized axis
	// and the angle (in radians).
	FastQuaternion(const FastVector3& axis, float angle)
	{
		_data = axis._data;
		// clear out W (scalar) component
		__m128 temp = _mm_setzero_ps();
		_data = _mm_insert_ps(_data, temp, 0x0);

		// load up sin (angle/2) and multiply the vector by this
		__m128 sinangleover2 = _mm_set_ps1(sinf(angle / 2.0f));
		_data = _mm_mul_ps(_data, sinangleover2);

		// now add in the scalar component
		__m128 scalar = _mm_set_ss(cosf(angle / 2.0f));
		_data = _mm_add_ps(_data, scalar);
	}

	// Constructs the quaternion given an __m128
	__forceinline FastQuaternion(__m128 value)
	{
		_data = value;
	}

	// Constructs a quaternion given the qv and qs components.
	// Note this assumes you have already applied the correct formula.
	FastQuaternion(float qv_x, float qv_y, float qv_z, float qs)
	{
		_data = _mm_set_ps(qv_x, qv_y, qv_z, qs);
	}

	// Copy constructor
	__forceinline FastQuaternion(const FastQuaternion& rhs)
	{
		_data = rhs._data;
	}

	// Assignment operator
	__forceinline FastQuaternion& operator=(const FastQuaternion& rhs)
	{
		_data = rhs._data;
		return *this;
	}

	// Returns the x component of the vector component (index 3)
	__forceinline float GetVectorX() const
	{
		return _data.m128_f32[3];
	}

	// Returns the y component of the vector component (index 2)
	__forceinline float GetVectorY() const
	{
		return _data.m128_f32[2];
	}

	// Returns the z component of the vector component (index 1)
	__forceinline float GetVectorZ() const
	{
		return _data.m128_f32[1];
	}

	// Returns the scalar component (index 0)
	__forceinline float GetScalar() const
	{
		return _data.m128_f32[0];
	}
	
	// Rotate by THIS quaternion, followed by rhs
	// Or in other words, p = rhs, q = this
	// Store result in this quaternion.
	void Multiply(const FastQuaternion& rhs)
	{
		// vector component is:
		// ps * qv + qs * pv + pv x qv
		
		// first calculate pv x qv
		
		// We effectively want to do this:
		// <A.y * B.z, A.z * B.x, A.x * B.y> - <A.z * B.y, A.x * B.z, A.y * B.x>
		// calculate first vector
		__m128 tempA = _mm_shuffle_ps(rhs._data, rhs._data, _MM_SHUFFLE(2, 1, 3, 0));
		__m128 tempB = _mm_shuffle_ps(_data, _data, _MM_SHUFFLE(1, 3, 2, 0));
		__m128 vector_component = _mm_mul_ps(tempA, tempB);

		// calculate second vector
		tempA = _mm_shuffle_ps(rhs._data, rhs._data, _MM_SHUFFLE(1, 3, 2, 0));
		tempB = _mm_shuffle_ps(_data, _data, _MM_SHUFFLE(2, 1, 3, 0));
		tempA = _mm_mul_ps(tempA, tempB);

		// finally, subtract the two vectors
		vector_component = _mm_sub_ps(vector_component, tempA);

		// calculate ps * qv, and add it to our vector component
		__m128 temp = _mm_shuffle_ps(rhs._data, rhs._data, _MM_SHUFFLE(0, 0, 0, 0));
		vector_component = _mm_add_ps(vector_component, _mm_mul_ps(_data, temp));

		// calculate qs * pv, and add it to our vector component
		__m128 temp2 = _mm_shuffle_ps(_data, _data, _MM_SHUFFLE(0, 0, 0, 0));
		vector_component = _mm_add_ps(vector_component, _mm_mul_ps(rhs._data, temp2));

		// scalar component is ps * qs - pv . qv
		// calculate ps * qs
		// ps is still in temp, qs is still in temp2
		temp = _mm_mul_ps(temp, temp2);

		// pv . qv
		__m128 scalar_component = _mm_dp_ps(rhs._data, _data, 0xE1);
		temp2 = _mm_set_ss(-1.0f);
		scalar_component = _mm_mul_ss(scalar_component, temp2);
		scalar_component = _mm_add_ss(scalar_component, temp); // this will leave 0s for X-Z

		// make sure our vector component has a W of 0
		vector_component = _mm_insert_ps(vector_component, vector_component, 0x1);

		// now combine the two components
		_data = _mm_add_ps(vector_component, scalar_component);
	}

	// Calculates the conjugate of this quaternion
	// Remember, for unit quaternions, the inverse is the conjugate.
	__forceinline void Conjugate()
	{
		__m128 temp = _mm_set_ps1(-1.0f);
		__m128 temp2 = _mm_set_ss(2.0f);
		temp = _mm_add_ss(temp, temp2);
		_data = _mm_mul_ps(_data, temp);
	}

	// Returns the length squared of this quaternion
	__forceinline float LengthSquared() const
	{
		const int Dot4Mask = 0xF1;
		__m128 result = _mm_dp_ps(_data, _data, Dot4Mask);
		return result.m128_f32[0];
	}

	// Returns the length of this quaternion
	__forceinline float Length() const
	{
		const int Dot4Mask = 0xF1;
		__m128 result = _mm_dp_ps(_data, _data, Dot4Mask);
		result = _mm_sqrt_ss(result);
		return result.m128_f32[0];
	}

	// Normalizes this quaternion
	__forceinline void Normalize()
	{
		// dot with self to get length squared
		const int Dot4Mask = 0xFF;
		__m128 result = _mm_dp_ps(_data, _data, Dot4Mask);

		// take reciprocal sqrt of dot product result 
		result = _mm_rsqrt_ps(result);

		// now multiply data by this
		_data = _mm_mul_ps(_data, result);
	}

	// Interpolates between quaternion a and b, returning the result by value.
	// Note that the result quaternion is also normalized before it is returned, to maintain
	// unit length.
	// result = normalize(a * (1.0f - f) + b * f)
	__forceinline friend FastQuaternion Lerp(const FastQuaternion& a, const FastQuaternion& b, float f)
	{
		__m128 pct = _mm_set_ps1(f);
		__m128 result = _mm_mul_ps(b._data, pct);
		__m128 ones = _mm_set_ps1(1.0f);
		pct = _mm_sub_ps(ones, pct);
		result = _mm_add_ps(result, _mm_mul_ps(a._data, pct));

		// now normalize the result
		// dot with self to get length squared
		const int Dot4Mask = 0xFF;
		__m128 dot = _mm_dp_ps(result, result, Dot4Mask);

		// take reciprocal sqrt of dot product result 
		dot = _mm_rsqrt_ps(dot);

		// now multiply data by this
		result = _mm_mul_ps(result, dot);

		return FastQuaternion(result);
	}

	// does a 4-way blend, normalizing the result
	// result = normalize(a * fa + b * fb + c * fc + d * fd)
	// fd = 1.0f - fa - fb - fc
	__forceinline friend FastQuaternion Blend(const FastQuaternion& a, const FastQuaternion& b, 
											  const FastQuaternion& c, const FastQuaternion& d,
											  float fa, float fb, float fc)
	{
		__m128 dpct = _mm_set_ps1(1.0f);
		__m128 pct = _mm_set_ps1(fa);
		__m128 result = _mm_mul_ps(a._data, pct);
		dpct = _mm_sub_ps(dpct, pct);

		pct = _mm_set_ps1(fb);
		__m128 temp = _mm_mul_ps(b._data, pct);
		result = _mm_add_ps(result, temp);
		dpct = _mm_sub_ps(dpct, pct);

		pct = _mm_set_ps1(fc);
		temp = _mm_mul_ps(c._data, pct);
		result = _mm_add_ps(result, temp);
		dpct = _mm_sub_ps(dpct, pct);

		temp = _mm_mul_ps(d._data, pct);
		result = _mm_add_ps(result, temp);

		// now normalize the result
		// dot with self to get length squared
		const int Dot4Mask = 0xFF;
		__m128 dot = _mm_dp_ps(result, result, Dot4Mask);

		// take reciprocal sqrt of dot product result 
		dot = _mm_rsqrt_ps(dot);

		// now multiply data by this
		result = _mm_mul_ps(result, dot);

		return FastQuaternion(result);
	}

	friend class FastVector3;
	friend class FastMatrix4;

	static const FastQuaternion Identity;
};

} // namespace Bits
#endif // _FASTMATH_H_
