Record Magma : Type := {
    carrier :> Type;
    op : carrier -> carrier -> carrier;
}.

Record Semigroup : Type := {
    magma :> Magma;
    op_assoc : forall x y z: carrier magma, op magma x (op magma y z) = op magma (op magma x y) z;
}.

Record Monoid : Type := {
    semigroup :> Semigroup;
    unit : carrier semigroup;
    op_unit_left : forall x : carrier semigroup, op semigroup unit x = x;
    op_unit_right : forall x : carrier semigroup, op semigroup x unit = x;
}.

Record Group : Type := {
    monoid :> Monoid;
    inv : carrier monoid -> carrier monoid;
    op_inv_left : forall x : carrier monoid, op monoid (inv x) x = unit monoid;
    op_inv_right : forall x : carrier monoid, op monoid x (inv x) = unit monoid;
}.

Record AbelianGroup : Type := {
    group :> Group;
    
    add := op group;
    oppos := inv group;
    zero := unit group;
    add_oppos_left := op_inv_left group;
    add_oppos_right := op_inv_right group;
    add_zero_left := op_unit_left group;
    add_zero_right := op_unit_right group;
    add_assoc := op_assoc group;

    add_comm : forall x y : carrier group, op group x y = op group y x;
}.

Record Ring : Type := {
    add_group :> AbelianGroup;

    mul : carrier add_group -> carrier add_group -> carrier add_group;
    mul_left_distrib : forall x y z : carrier add_group, 
        mul (add add_group x y) z = add add_group (mul x z) (mul y z);
    mul_right_distrib : forall x y z : carrier add_group, 
        mul x (add add_group y z) = add add_group (mul x y) (mul x z);
}.

Record AssociativeRing : Type := {
    ring :> Ring;
    mul_assoc : forall x y z : carrier ring,
        mul ring (mul ring x y) z = mul ring x (mul ring y z);
}.

Record AssociativeUnitaryRing : Type := {
    assoc_ring :> AssociativeRing;
    mul_unit : carrier assoc_ring;
    mul_unit_left : forall x : carrier assoc_ring, mul assoc_ring mul_unit x = x;
    mul_unit_right : forall x : carrier assoc_ring, mul assoc_ring x mul_unit = x;
}.
