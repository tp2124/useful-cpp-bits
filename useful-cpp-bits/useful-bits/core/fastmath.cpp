// FastMath.cpp defines statics for the fast math library
#include "fastmath.h"

namespace Bits
{
float _ident[4][4] = {1.0f, 0.0f, 0.0f, 0.0f,
						0.0f, 1.0f, 0.0f, 0.0f,
						0.0f, 0.0f, 1.0f, 0.0f,
						0.0f, 0.0f, 0.0f, 1.0f};
const FastMatrix4 FastMatrix4::Identity(_ident);

const FastVector3 FastVector3::Zero(0.0f, 0.0f, 0.0f);
const FastVector3 FastVector3::UnitX(1.0f, 0.0f, 0.0f);
const FastVector3 FastVector3::UnitY(0.0f, 1.0f, 0.0f);
const FastVector3 FastVector3::UnitZ(0.0f, 0.0f, 1.0f);
const FastVector3 FastVector3::NegativeUnitX(-1.0f, 0.0f, 0.0f);
const FastVector3 FastVector3::NegativeUnitY(0.0f, -1.0f, 0.0f);
const FastVector3 FastVector3::NegativeUnitZ(0.0f, 0.0f, -1.0f);

const FastQuaternion FastQuaternion::Identity(0.0f, 0.0f, 0.0f, 1.0f);

void FastMatrix4::CreateTranslation(const FastVector3& translation)
{
	__m128 temp = translation._data;

	// set W component to 0.0f before we do anything
	temp = _mm_insert_ps(temp, temp, 0x01);

	__m128 one = _mm_set_ss(1.0f);

	// 1 0 0 temp.x
	_rows[0] = _mm_shuffle_ps(temp, one, _MM_SHUFFLE(0, 1, 0, 3));

	// 0 1 0 temp.y
	_rows[1] = _mm_shuffle_ps(temp, one, _MM_SHUFFLE(1, 0, 0, 2));

	// now set W to 1.0f
	temp = _mm_max_ss(temp, one);

	// 0 0 1 temp.z
	_rows[2] = _mm_shuffle_ps(temp, one, _MM_SHUFFLE(1, 1, 0, 1));

	// 0 0 0 1
	_rows[3] = one;
}

void FastMatrix4::CreateFromQuaternion(const FastQuaternion& q)
{
// 	float x = q.GetVectorX();
// 	float y = q.GetVectorY();
// 	float z = q.GetVectorZ();
// 	float w = q.GetScalar();
// 	
// 	_rows[0] = _mm_set_ps(1.0f-2.0f*y*y-2.0f*z*z, 2.0f*x*y-2.0f*z*w, 2.0f*x*z+2.0f*y*w, 0.0f);
// 	_rows[1] = _mm_set_ps(2.0f*x*y+2.0f*z*w, 1.0f-2.0f*x*x-2.0f*z*z, 2.0f*y*z-2.0f*x*w, 0.0f);
// 	_rows[2] = _mm_set_ps(2.0f*x*z-2.0f*y*w, 2.0f*y*z+2.0f*x*w, 1.0f-2.0f*x*x-2.0f*y*y, 0.0f);
// 	_rows[3] = _mm_set_ss(1.0f);
 	__declspec(align(16)) static const float negmsk1[4] = {0.0f, 1.0f, 1.0f, -1.0f};
 	__declspec(align(16)) static const float negmsk2[4] = {0.0f, 1.0f, -1.0f, -1.0f};
 	__declspec(align(16)) static const float negmsk3[4] = {0.0f, 1.0f, -1.0f, 1.0f};
 	__declspec(align(16)) static const float negmsk4[4] = {0.0f, -1.0f, -1.0f, 1.0f};
 	__declspec(align(16)) static const float negmsk5[4] = {0.0f, -1.0f, 1.0f, 1.0f};
 	__declspec(align(16)) static const float negmsk6[4] = {0.0f, -1.0f, 1.0f, -1.0f};
 	
 	__m128 x_perms = _mm_shuffle_ps(q._data, q._data, _MM_SHUFFLE(3, 3, 3, 3)); // splat q.data.x
 	x_perms = _mm_mul_ps(x_perms, q._data);
 	x_perms = _mm_add_ps(x_perms, x_perms); // 2*x*x, 2*x*y, 2*x*z, 2*x*w
 	
 	__m128 y_perms = _mm_shuffle_ps(q._data, q._data, _MM_SHUFFLE(2, 2, 2, 2)); // splat q.data.y
 	y_perms = _mm_mul_ps(y_perms, q._data);
 	y_perms = _mm_add_ps(y_perms, y_perms); // 2*y*x, 2*y*y, 2*y*z, 2*y*w
 
 	__m128 z_perms = _mm_shuffle_ps(q._data, q._data, _MM_SHUFFLE(1, 1, 1, 1)); // splat q.data.z
 	z_perms = _mm_mul_ps(z_perms, q._data);
 	z_perms = _mm_add_ps(z_perms, z_perms); // 2*z*x, 2*z*y, 2*z*z, 2*z*w
 
 	__m128 negmask;
 	__m128 temp;
 	// construct rows[0]
 	// <1, 0, 0> + <-2*y*y, 2*x*y, 2*x*z> + <-2*z*z, -2*z*w, 2*y*w>
 	_rows[0] = _mm_set_ss(1.0f);
 	_rows[0] = _mm_shuffle_ps(_rows[0], _rows[0], _MM_SHUFFLE(0, 1, 1, 1));
 	
 	temp = _mm_shuffle_ps(x_perms, y_perms, _MM_SHUFFLE(2, 3, 1, 0));
 	negmask = _mm_load_ps(negmsk1);
 	temp = _mm_mul_ps(temp, negmask);
 	_rows[0] = _mm_add_ps(_rows[0], temp);
 
 	temp = _mm_shuffle_ps(y_perms, z_perms, _MM_SHUFFLE(1, 0, 0, 0));
 	negmask = _mm_load_ps(negmsk2);
 	temp = _mm_mul_ps(temp, negmask);
 	_rows[0] = _mm_add_ps(_rows[0], temp);
 
 	// construct rows[1]
 	// <0, 1, 0> + <2*x*y, -2*x*x, 2*y*z> + <2*z*w, -2*z*z, -2*x*w>
 	_rows[1] = _mm_set_ss(1.0f);
 	_rows[1] = _mm_shuffle_ps(_rows[1], _rows[1], _MM_SHUFFLE(1, 0, 1, 1));
 
 	temp = _mm_shuffle_ps(y_perms, x_perms, _MM_SHUFFLE(2, 3, 1, 0));
 	negmask = _mm_load_ps(negmsk3);
 	temp = _mm_mul_ps(temp, negmask);
 	_rows[1] = _mm_add_ps(_rows[1], temp);
 
 	temp = _mm_shuffle_ps(x_perms, z_perms, _MM_SHUFFLE(0, 1, 0, 0));
 	negmask = _mm_load_ps(negmsk4);
 	temp = _mm_mul_ps(temp, negmask);
 	_rows[1] = _mm_add_ps(_rows[1], temp);
 
 	// construct rows[2]
 	// <0, 0, 1> + <2*x*z, 2*y*z, -2*x*x> + <-2*y*w, 2*x*w, -2*y*y>
 	_rows[2] = _mm_set_ss(1.0f);
 	_rows[2] = _mm_shuffle_ps(_rows[2], _rows[2], _MM_SHUFFLE(1, 1, 0, 1));
 
 	temp = _mm_shuffle_ps(x_perms, z_perms, _MM_SHUFFLE(3, 2, 3, 0));
 	negmask = _mm_load_ps(negmsk5);
 	temp = _mm_mul_ps(temp, negmask);
 	_rows[2] = _mm_add_ps(_rows[2], temp);
 
 	temp = _mm_unpacklo_ps(y_perms, x_perms); // 2*x*z, 2*y*z, 2*x*w, 2*y*w
 	temp = _mm_shuffle_ps(y_perms, temp, _MM_SHUFFLE(0, 1, 2, 0));
 	negmask = _mm_load_ps(negmsk6);
 	temp = _mm_mul_ps(temp, negmask);
 	_rows[2] = _mm_add_ps(_rows[2], temp);
 
 	// construct rows[3]
 	_rows[3] = _mm_set_ss(1.0f);
}

// Constructs a Look-At matrix
// vUp MUST be normalized or bad things will happen
void FastMatrix4::CreateLookAt( const FastVector3& vEye, const FastVector3& vAt, const FastVector3& vUp )
{
	FastVector3 vFront = vAt;
	vFront.Sub(vEye);
	vFront.Normalize();

	FastVector3 vLeft = Cross(vUp, vFront);
	vLeft.Normalize();

	FastVector3 vNewUp = Cross(vFront, vLeft);
	vNewUp.Normalize();

	__m128 negate = _mm_set_ss(-1.0f);
	const int Dot3Mask = 0xE1;
	_rows[0] = vLeft._data;
	__m128 temp = _mm_dp_ps(vEye._data, _rows[0], Dot3Mask);
	temp = _mm_mul_ps(temp, negate);
	_rows[0] = _mm_insert_ps(_rows[0], temp, 0x0);
	
	_rows[1] = vNewUp._data;
	temp = _mm_dp_ps(vEye._data, _rows[1], Dot3Mask);
	temp = _mm_mul_ps(temp, negate);
	_rows[1] = _mm_insert_ps(_rows[1], temp, 0x0);

	_rows[2] = vFront._data;
	temp = _mm_dp_ps(vEye._data, _rows[2], Dot3Mask);
	temp = _mm_mul_ps(temp, negate);
	_rows[2] = _mm_insert_ps(_rows[2], temp, 0x0);

	_rows[3] = _mm_set_ps(0.0f, 0.0f, 0.0f, 1.0f);
}

void FastMatrix4::CreatePerspectiveFOV(float fFOVy, float fAspectRatio, float fNear, float fFar)
{
	float fYScale = tanf(1.57079633f - (fFOVy/2)); // cot(x) is the same as tan(pi/2 - x)
	float fXScale = fYScale / fAspectRatio;

	_rows[0] = _mm_set_ps(fXScale, 0.0f, 0.0f, 0.0f);
	_rows[1] = _mm_set_ps(0.0f, fYScale, 0.0f, 0.0f);
	_rows[2] = _mm_set_ps(0.0f, 0.0f, fFar/(fFar-fNear), -fNear*fFar/(fFar-fNear));
	_rows[3] = _mm_set_ps(0.0f, 0.0f, 1.0f, 0.0f);
}

void FastVector3::Rotate(const FastQuaternion& q)
{
	// v + 2.0*cross(q.xyz, cross(q.xyz,v) + q.w*v);

	// set W component to 0.0f before we do anything
	_data = _mm_insert_ps(_data, _data, 0x01);

	// first calculate q x v
	__m128 temp_q = q._data;
	temp_q = _mm_insert_ps(temp_q, temp_q, 0x01);

	// calculate first vector
	__m128 tempA = _mm_shuffle_ps(temp_q, temp_q, _MM_SHUFFLE(2, 1, 3, 0));
	__m128 tempB = _mm_shuffle_ps(_data, _data, _MM_SHUFFLE(1, 3, 2, 0));
	__m128 result = _mm_mul_ps(tempA, tempB);

	// calculate second vector
	tempA = _mm_shuffle_ps(temp_q, temp_q, _MM_SHUFFLE(1, 3, 2, 0));
	tempB = _mm_shuffle_ps(_data, _data, _MM_SHUFFLE(2, 1, 3, 0));
	tempA = _mm_mul_ps(tempA, tempB);

	// finally, subtract the two vectors
	result = _mm_sub_ps(result, tempA);

	// now calculate q.w * v
	__m128 temp1 = _mm_shuffle_ps(q._data, q._data, _MM_SHUFFLE(0, 0, 0, 0));
	temp1 = _mm_mul_ps(_data, temp1);

	// result + temp1
	result = _mm_add_ps(result, temp1);

	// now do q x result
	// calculate first vector
	tempA = _mm_shuffle_ps(temp_q, temp_q, _MM_SHUFFLE(2, 1, 3, 0));
	tempB = _mm_shuffle_ps(result, result, _MM_SHUFFLE(1, 3, 2, 0));
	__m128 partial1 = _mm_mul_ps(tempA, tempB);

	// calculate second vector
	tempA = _mm_shuffle_ps(temp_q, temp_q, _MM_SHUFFLE(1, 3, 2, 0));
	tempB = _mm_shuffle_ps(result, result, _MM_SHUFFLE(2, 1, 3, 0));
	__m128 partial2 = _mm_mul_ps(tempA, tempB);

	result = _mm_sub_ps(partial1, partial2);

	// now multiply by 2
	temp1 = _mm_set_ps1(2.0f);
	result = _mm_mul_ps(result, temp1);

	// now add v to it to get the actual transformed value
	_data = _mm_add_ps(_data, result);
}

} // namespace Bits
