from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class RelationshipType(str, Enum):
    FATHER = "FATHER"
    MOTHER = "MOTHER"
    SPOUSE = "SPOUSE"
    CHILD = "CHILD"
    BROTHER = "BROTHER"
    SISTER = "SISTER"
    PATERNAL_GRANDFATHER = "PATERNAL_GRANDFATHER"
    PATERNAL_GRANDMOTHER = "PATERNAL_GRANDMOTHER"
    MATERNAL_GRANDFATHER = "MATERNAL_GRANDFATHER"
    MATERNAL_GRANDMOTHER = "MATERNAL_GRANDMOTHER"
    GRANDCHILD = "GRANDCHILD"
    PATERNAL_UNCLE = "PATERNAL_UNCLE"
    PATERNAL_AUNT = "PATERNAL_AUNT"
    MATERNAL_UNCLE = "MATERNAL_UNCLE"
    MATERNAL_AUNT = "MATERNAL_AUNT"
    NIBLING = "NIBLING"
    PATERNAL_UNCLE_CHILD = "PATERNAL_UNCLE_CHILD"
    PATERNAL_AUNT_CHILD = "PATERNAL_AUNT_CHILD"
    MATERNAL_UNCLE_CHILD = "MATERNAL_UNCLE_CHILD"
    MATERNAL_AUNT_CHILD = "MATERNAL_AUNT_CHILD"
    FATHER_IN_LAW = "FATHER_IN_LAW"
    MOTHER_IN_LAW = "MOTHER_IN_LAW"
    OLDER_BROTHER_IN_LAW = "OLDER_BROTHER_IN_LAW"
    YOUNGER_BROTHER_IN_LAW = "YOUNGER_BROTHER_IN_LAW"
    OLDER_SISTER_IN_LAW = "OLDER_SISTER_IN_LAW"
    YOUNGER_SISTER_IN_LAW = "YOUNGER_SISTER_IN_LAW"
    SON_IN_LAW = "SON_IN_LAW"
    DAUGHTER_IN_LAW = "DAUGHTER_IN_LAW"
    GUARDIAN = "GUARDIAN"
    WARD = "WARD"
    OTHER = "OTHER"


class RelativeInfo(BaseModel):
    relation: Optional[RelationshipType] = Field(None, description="Qarindoshlik turi")
    full_name: Optional[str] = Field(None, description="To'liq ismi sharifi")
    birth_year_place: Optional[str] = Field(None, description="Tug'ilgan yili va joyi")
    workplace_position: Optional[str] = Field(None, description="Ish joyi va lavozimi")
    address: Optional[str] = Field(None, description="Turar joyi")


class WorkHistory(BaseModel):
    period: Optional[str] = Field(None, description="Davr (masalan: 1993-1997 yy.)")
    description: Optional[str] = Field(None, description="Ish yoki o'qish joyi va lavozimi")


class ObyektivkaData(BaseModel):
    surname: Optional[str] = Field(None, description="Familiyasi")
    name: Optional[str] = Field(None, description="Ismi")
    patronymic: Optional[str] = Field(None, description="Otasining ismi (sharifi)")
    current_position: Optional[str] = Field(None, description="Hozirgi lavozimi va ish joyi")
    position_since: Optional[str] = Field(None, description="Lavozimga qachondan ishlaydi")
    birth_date: Optional[str] = Field(None, description="Tug'ilgan sanasi")
    birth_place: Optional[str] = Field(None, description="Tug'ilgan joyi")
    nationality: Optional[str] = Field(None, description="Millati")
    party_membership: Optional[str] = Field(None, description="Partiyaviylik")
    education_level: Optional[str] = Field(None, description="Ma'lumoti (oliy, o'rta va h.k.)")
    graduated_from: Optional[str] = Field(None, description="Qayerda va qachon tahsil olgan")
    specialty: Optional[str] = Field(None, description="Mutaxassisligi")
    academic_degree: Optional[str] = Field(None, description="Ilmiy darajasi")
    academic_title: Optional[str] = Field(None, description="Ilmiy unvoni")
    foreign_languages: Optional[str] = Field(None, description="Biladigan chet tillari")
    state_awards: Optional[str] = Field(None, description="Davlat mukofotlari")
    deputy_status: Optional[str] = Field(None, description="Deputatlik holati")
    work_history: List[WorkHistory] = Field(default_factory=list, description="Mehnat faoliyati tarixi")
    relatives: List[RelativeInfo] = Field(default_factory=list, description="Yaqin qarindoshlari haqida ma'lumot")


class ObyektivkaResponse(BaseModel):
    success: bool
    filename: str
    file_type: str
    data: ObyektivkaData
    raw_text_preview: Optional[str] = Field(None, description="Fayldan o'qilgan matnning qisqa ko'rinishi")
