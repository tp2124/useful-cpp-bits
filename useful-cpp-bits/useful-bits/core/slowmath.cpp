// SlowMath.cpp defines statics for the slow math library
#include "slowmath.h"

namespace Bits
{
float _ident2[4][4] = {1.0f, 0.0f, 0.0f, 0.0f,
						0.0f, 1.0f, 0.0f, 0.0f,
						0.0f, 0.0f, 1.0f, 0.0f,
						0.0f, 0.0f, 0.0f, 1.0f};
const SlowMatrix4 SlowMatrix4::Identity(_ident2);

const SlowVector3 SlowVector3::Zero(0.0f, 0.0f, 0.0f);
const SlowVector3 SlowVector3::UnitX(1.0f, 0.0f, 0.0f);
const SlowVector3 SlowVector3::UnitY(0.0f, 1.0f, 0.0f);
const SlowVector3 SlowVector3::UnitZ(0.0f, 0.0f, 1.0f);
const SlowVector3 SlowVector3::NegativeUnitX(-1.0f, 0.0f, 0.0f);
const SlowVector3 SlowVector3::NegativeUnitY(0.0f, -1.0f, 0.0f);
const SlowVector3 SlowVector3::NegativeUnitZ(0.0f, 0.0f, -1.0f);

const SlowQuaternion SlowQuaternion::Identity(0.0f, 0.0f, 0.0f, 1.0f);

void SlowMatrix4::CreateTranslation(const SlowVector3& rhs)
{
	memset(_matrix, 0, sizeof(float) * 16);
	_matrix[0][0] = 1.0f;
	_matrix[0][3] = rhs._x;
	_matrix[1][1] = 1.0f;
	_matrix[1][3] = rhs._y;
	_matrix[2][2] = 1.0f;
	_matrix[2][3] = rhs._z;
	_matrix[3][3] = 1.0f;
}

void SlowMatrix4::CreateFromQuaternion(const SlowQuaternion& q)
{
	float x = q.GetVectorX();
	float y = q.GetVectorY();
	float z = q.GetVectorZ();
	float w = q.GetScalar();

	//rows[0] = _mm_set_ps(1.0f-2.0f*y*y-2.0f*z*z, 2.0f*x*y-2.0f*z*w, 2.0f*x*z+2.0f*y*w, 0.0f);
	//rows[1] = _mm_set_ps(2.0f*x*y+2.0f*z*w, 1.0f-2.0f*x*x-2.0f*z*z, 2.0f*y*z-2.0f*x*w, 0.0f);
	//rows[2] = _mm_set_ps(2.0f*x*z-2.0f*y*w, 2.0f*y*z+2.0f*x*w, 1.0f-2.0f*x*x-2.0f*y*y, 0.0f);
	//rows[3] = _mm_set_ss(1.0f);

	_matrix[0][0] = 1.0f-2.0f*y*y-2.0f*z*z; 
	_matrix[0][1] = 2.0f*x*y-2.0f*z*w;
	_matrix[0][2] = 2.0f*x*z+2.0f*y*w;
	_matrix[0][3] = 0.0f;

	_matrix[1][0] = 2.0f*x*y+2.0f*z*w;
	_matrix[1][1] = 1.0f-2.0f*x*x-2.0f*z*z;
	_matrix[1][2] = 2.0f*y*z-2.0f*x*w;
	_matrix[1][3] = 0.0f;

	_matrix[2][0] = 2.0f*x*z-2.0f*y*w;
	_matrix[2][1] = 2.0f*y*z+2.0f*x*w;
	_matrix[2][2] = 1.0f-2.0f*x*x-2.0f*y*y;
	_matrix[2][3] = 0.0f;

	_matrix[3][0] = 0.0f;
	_matrix[3][1] = 0.0f;
	_matrix[3][2] = 0.0f;
	_matrix[3][3] = 1.0f;
}

void SlowMatrix4::CreateLookAt( const SlowVector3& vEye, const SlowVector3& vAt, const SlowVector3& vUp )
{
	SlowVector3 vFront = vAt;
	vFront.Sub(vEye);
	vFront.Normalize();

	SlowVector3 vLeft = Cross(vUp, vFront);
	vLeft.Normalize();

	SlowVector3 vNewUp = Cross(vFront, vLeft);
	vNewUp.Normalize();

	_matrix[0][0] = vLeft.GetX();
	_matrix[0][1] = vLeft.GetY();
	_matrix[0][2] = vLeft.GetZ();
	_matrix[0][3] = -1.0f * vLeft.Dot(vEye);

	_matrix[1][0] = vNewUp.GetX();
	_matrix[1][1] = vNewUp.GetY();
	_matrix[1][2] = vNewUp.GetZ();
	_matrix[1][3] = -1.0f * vNewUp.Dot(vEye);

	_matrix[2][0] = vFront.GetX();
	_matrix[2][1] = vFront.GetY();
	_matrix[2][2] = vFront.GetZ();
	_matrix[2][3] = -1.0f * vFront.Dot(vEye);

	_matrix[3][0] = 0.0f;
	_matrix[3][1] = 0.0f;
	_matrix[3][2] = 0.0f;
	_matrix[3][3] = 1.0f;
}

void SlowMatrix4::CreatePerspectiveFOV( float fFOVy, float fAspectRatio, float fNear, float fFar )
{
	float fYScale = tanf(1.57079633f - (fFOVy/2)); // cot(x) is the same as tan(pi/2 - x)
	float fXScale = fYScale / fAspectRatio;

	memset(_matrix, 0, sizeof(float) * 16);
	_matrix[0][0] = fXScale;
	_matrix[1][1] = fYScale;
	_matrix[2][2] = fFar/(fFar-fNear);
	_matrix[2][3] = -fNear*fFar/(fFar-fNear);
	_matrix[3][2] = 1.0f;
}

void SlowVector3::Rotate(const SlowQuaternion& q)
{
	// v + 2.0*cross(q.xyz, cross(q.xyz,v) + q.w*v);
	SlowVector3 temp(*this);
	temp.Multiply(q.GetScalar());
	temp.Add(Cross(q._qv, *this));

	temp = Cross(q._qv, temp);
	temp.Multiply(2.0f);
	temp.Add(*this);

	_x = temp._x;
	_y = temp._y;
	_z = temp._z;
	_w = temp._w;
}

} // namespace Bits
