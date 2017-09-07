#define	G	1.0f

typedef struct __attribute__ ((packed)) _body_movement_state
{
	float3 pos;
	float3 vel;
};

typedef struct __attribute__ ((packed)) _body_properties
{
	float mass;
	float radius;
	float coll_coef;
};

__kernel void n_body(__global struct _body_movement_state * bodies_mstate_old,	// movement status of the bodies (in)
					 __global struct _body_properties * bodies_props,			// properties of the bodies	(in)
					 int count,													// number of the bodies (in)
					 float dt,													// time step (in)
					 __global struct _body_movement_state * bodies_mstate_new)	// new movement status (out)
{
	int a = get_global_id(0);
	
	if(a < count){
		struct _body_movement_state mov_body_a = bodies_mstate_old[a];
		struct _body_properties prop_body_a = bodies_props[a];

		int b;
		float3 F = (float3)(0.0f, 0.0f, 0.0f);	// net force of gravity
		float3 u = (float3)(0.0f, 0.0f, 0.0f);	// velocity after collisions
		bool collision = false;	// collision happened
		for(b = 0; b < count; b++){
			/* if it's body A */
			if(a == b)
				continue;

			/* body B */
			struct _body_movement_state mov_body_b = bodies_mstate_old[b];
			struct _body_properties prop_body_b = bodies_props[b];

			float3 diff = mov_body_b.pos -  mov_body_a.pos;		// difference vector (also the direction of the grav force)
			float dist = sqrt((diff.x * diff.x) + (diff.y * diff.y) + (diff.z * diff.z));	// distance is the magnitude of the diff. vector
			float3 unit_diff = diff * (1 / dist);	// normalized diff for the direction of grav force

			float3 F_G = ((G * prop_body_a.mass *  prop_body_b.mass) / (dist * dist)) * unit_diff;	// Newton's grav law

			F += F_G;	// system is linear -> superposition -> simply sum the forces

			/* collision */
			if(dist <= (prop_body_a.radius + prop_body_b.radius)){
				/*  elastic collision 
				   I : v_a + u_a = v_b + u_b
				  II : m_a * v_a + m_b * v_b = m_a * u_a + m_b * u_b */
				
				float3 u_a = (prop_body_a.mass * mov_body_a.vel + 2.0f * prop_body_b.mass * mov_body_b.vel - prop_body_b.mass * mov_body_a.vel) / ( 2 * prop_body_a.mass);

				u += u_a;
				
				collision = true;
			}
		}
	 
		float3 acc = F / prop_body_a.mass;	// Newton's II. law
		
		float3 v = collision ? u : mov_body_a.vel;

		mov_body_a.vel = v + (acc * dt);
		mov_body_a.pos += (mov_body_a.vel * dt);
		
		bodies_mstate_new[a] = mov_body_a;	// register the changed status
	}
}