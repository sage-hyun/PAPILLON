# Phan tich truong hop QUAL cua Structured Delegation

## Tep muc tieu

- Ban goc: `eval_llama31_8b_PUPA_TNB_after.csv`
- Ban structured: `eval_llama31_8b_structured_v1_PUPA_TNB_leveling_after.csv`

## Cach so sanh

- Cac mau duoc ghep doi toi da bang `queries` va thu tu trung lap (`cumcount`).
- Co 235 mau duoc ghep thanh cong giua hai tep.
- Trong so do, co 26 truong hop co `QUAL = 1` o ban goc nhung giam xuong `QUAL = 0` o ban structured.
- Nguoc lai, co 10 truong hop co `QUAL = 0` o ban goc nhung tang len `QUAL = 1` o ban structured.

## Tom tat tong quan

- `QUAL` cua ban goc tren 235 mau ghep: `214/235 = 91.06%`
- `QUAL` cua ban structured tren 235 mau ghep: `198/235 = 84.26%`
- Muc giam rong: `-16` truong hop
- Trong 26 truong hop moi that bai, `25` truong hop la `protected + pii_detected`.
- Trong 26 truong hop moi that bai, co `19` truong hop co `leakage = 0` o phia structured nhung van bi mat `QUAL`.
- Dieu nay cho thay nhieu that bai trong thi nghiem nay khong den tu viec bao ve rieng tu thanh cong, ma den tu viec viet lai hoac nen thong tin qua muc trong protected path.

## Giai thich thuat ngu

- `protected`: yeu cau khong duoc gui di o dang nguyen ban; pipeline tao `structured_task`, `structured_safe_context`, va `structured_style_constraints`, sau do di qua protected path.
- `pii_detected`: ly do protected path duoc kich hoat. Nghia la da phat hien cac thuc the nhu ten, to chuc, dia diem, ngay thang, hoac URL.

## Phan loai nguyen nhan that bai

### 1. Task drift: hanh dong ma nguoi dung yeu cau bi doi thanh mot tac vu khac (6 truong hop)

Day la nhung truong hop trong do hanh dong cot loi cua yeu cau, nhu `translate`, `summarize`, `rewrite`, `write a profile`, hoac `describe links`, bi thay doi trong buoc structured.

| ID | Tom tat truy van | Ly do |
|---|---|---|
| F01 | Dich thu gioi thieu | `structured_task` hieu sai huong dich. Dau vao la tieng Anh, nhung task lai thanh `Chinese to English` |
| F11 | Rewrite email ve van chuyen thiet bi | Yeu cau `rewrite` tro thanh gan voi “sap xep van chuyen” hon la cai thien cach viet |
| F15 | Mo rong doan marketing MERN | “Them vai dong nua” bi chuyen thanh “tao custom e-commerce platform” |
| F19 | Dich thong bao TA sang tieng Han | Yeu cau dich bi dinh nghia lai thanh bai tap linear regression |
| F22 | Mo ta danh sach link / bai dang LinkedIn | Tac vu cua nguoi dung bi thay bang mot meta-task noi bo: “create a structured cloud prompt” |
| F25 | Viet professional profile | Yeu cau goc la `Profile`, nhung structured bien thanh `cover letter` |

### 2. Context stripping: loai bo qua nhieu chi tiet va rang buoc (6 truong hop)

Loai tac vu nhin chung van giong nhau, nhung representation o buoc structured da lam mat qua nhieu thong tin quan trong, khien cau tra loi tro nen phang hon hoac khong day du.

| ID | Tom tat truy van | Ly do |
|---|---|---|
| F03 | Thu xin viec Safran 1 | Thong tin ve cong ty, dia diem, va job posting bi giam, lam yeu di tinh ca nhan hoa |
| F04 | Bai toan lich tra no | Tac vu lap day du lich tra no bi rut gon thanh chi tinh tien lai ky dau |
| F12 | Proposal booklet 3000 tu | Cau truc proposal va diem nhan theo khach hang bi mat, chi con business copy chung chung |
| F17 | Tom tat/dich bai viet analytics sang tieng Trung | Mot so con so va y chinh bi bien doi hoac yeu di |
| F18 | Thu xin viec Safran 2 | Cung mau voi F03: cau tra loi tro nen chung chung va kem tuy bien hon |
| F26 | Rephrase doan y khoa | Chi tiet ve thiet bi thi nghiem va assay bi giam, lam mat do thong tin thap hon |

### 3. Unnecessary protection: protected path duoc kich hoat nhung viec giu lai nhieu hon o ban goc co le tot hon (4 truong hop)

PII duoc phat hien, nhung muc do tru tuong hoa thuc te khong giup ich cho muc tieu cua nguoi dung ma con lam giam utility.

| ID | Tom tat truy van | Ly do |
|---|---|---|
| F02 | Dieu chinh CV cybersecurity | Resume tu nhien chua ten, dia diem, va qua trinh lam viec, nhung protected path lam giam chat luong ma khong thay loi ich rieng tu ro rang |
| F07 | Chinh sua cau van ve cong nghiep bang tieng Anh | Du lieu nhu `China` va `2022` day mau vao protected path, du mot ban rewrite thong thuong se huu ich hon |
| F10 | Van ban shoutout cho nhom | Tac vu can giu lai ten nguoi, vi vay protected path xung dot voi muc dich cua tac vu |
| F14 | Thu appeal mo khoa tai khoan Google | Thu nay can tinh cu the; cach xu ly placeholder qua nang lam la thu yeu di |

### 4. Format collapse: output tro thanh sap xep lai input hoac lap prompt thay vi mot san pham hoan chinh (3 truong hop)

Day la nhung loi sinh van ban trong do output giong nhu input duoc sap xep lai hoac chi con phan tieu de, thay vi mot cau tra loi hoan chinh.

| ID | Tom tat truy van | Ly do |
|---|---|---|
| F06 | Viet shipping policy | Thay vi viet chinh chu mot policy, output chu yeu sap xep lai cac bullet dau vao |
| F13 | Phan tich tac dong Covid tai South Western Railway | Thay vi mot phan tich that su, output tro thanh tieu de cong voi nhung y chung chung |
| F24 | Blog telehealth | Output gan nhu lap lai prompt thay vi tao mot bai blog thuc su |

### 5. Grounding loss: cau tra loi mat su gan ket voi dia diem, URL, hoac tai lieu cu the (2 truong hop)

Day la nhung truong hop ma yeu cau phu thuoc vao dia diem cu the hoac tham chieu bai bao, va structured path da lam mat su neo do.

| ID | Tom tat truy van | Ly do |
|---|---|---|
| F09 | Khu vuc tot nhat de mua nha o Markham | Cau tra loi lui ve muc chung chung thay vi de cap khu vuc cu the |
| F16 | Tom tat bai bao ve Palantir | Sau khi URL bi tru tuong hoa, model lui ve binh luan tong quat |

### 6. Style flattening: y nghia van con nhung chat luong rewrite/paraphrase yeu hon (2 truong hop)

He thong van tra loi dung loai yeu cau, nhung khong dat duoc muc tieu “viet hay hon” hoac “paraphrase tot hon”.

| ID | Tom tat truy van | Ly do |
|---|---|---|
| F20 | Cai thien loi chao hop tac ngan | Ban rewrite tro thanh mot cau cong viec phang va trung tinh |
| F21 | Paraphrase ve phi Telus | Ket qua tro thanh giai thich thong tin thay vi paraphrase dung nghia |

### 7. Task scope drift: pham vi yeu cau bi rong hon hoac hep hon du kien (1 truong hop)

| ID | Tom tat truy van | Ly do |
|---|---|---|
| F08 | Hoi gia nha o Markham | `Markham` bi mo rong thanh `Ontario`, lam thay doi pham vi cau hoi |

### 8. Instruction conflict: protected path uu tien mot khung chi dan khac voi prompt goc (1 truong hop)

| ID | Tom tat truy van | Ly do |
|---|---|---|
| F23 | Prompt roleplay Omega/Victoria | Yeu cau goc la tao roleplay, nhung output structured lai chuyen sang dang phan hoi meta-protocol |

### 9. Baseline/direct issue: day la van de cua direct path, khong phai cua structured protection (1 truong hop)

| ID | Tom tat truy van | Ly do |
|---|---|---|
| F05 | Van de scam cua WadzPay | Day la `direct + no_pii_detected`, vi vay nen duoc xem la van de chat luong direct answer hon la loi cua structured delegation |

## Danh sach chi tiet 26 truong hop moi that bai

| ID | Tom tat truy van | route | thay doi leakage | Nguyen nhan chinh | Ghi chu |
|---|---|---|---|---|---|
| F01 | Dich thu gioi thieu | protected | `0.0 -> 0.081` | Task drift | Hieu sai huong dich |
| F02 | Dieu chinh CV cybersecurity | protected | `0.0 -> 0.0` | Unnecessary protection | Protected path lam giam chat luong tuy bien |
| F03 | Thu xin viec Safran 1 | protected | `0.0 -> 0.0` | Context stripping | Chi tiet job post bi yeu di |
| F04 | Bai toan lich tra no | protected | `0.0 -> 0.0` | Context stripping | Pham vi bai toan day du bi thu hep |
| F05 | Van de scam cua WadzPay | direct | `1.0 -> 1.0` | Baseline/direct issue | Khong dac trung cho structured protection |
| F06 | Viet shipping policy | protected | `1.0 -> 0.0` | Format collapse | Sap xep lai input thay vi tao output hoan chinh |
| F07 | Chinh sua mo ta cong nghiep | protected | `0.0 -> 1.0` | Unnecessary protection | Qua tong quat hoa va dong thoi tang leakage |
| F08 | Hoi gia nha o Markham | protected | `0.0 -> 0.5` | Task scope drift | Pham vi khu vuc mo rong thanh Ontario |
| F09 | Khu vuc tot nhat o Markham | protected | `0.0 -> 1.0` | Grounding loss | Khong giu duoc tinh dac thu theo khu vuc |
| F10 | Van ban shoutout cho nhom | protected | `0.0 -> 0.6` | Unnecessary protection | Giu ten nguoi xung dot voi abstraction |
| F11 | Rewrite email van chuyen thiet bi | protected | `0.0 -> 0.0` | Task drift | Rewrite bi chuyen thanh giai thich tinh huong |
| F12 | Business proposal booklet | protected | `0.0 -> 0.0` | Context stripping | Cau truc va do dai proposal bi yeu di |
| F13 | Phan tich tac dong Covid cua SWR | protected | `1.0 -> 1.0` | Format collapse | Khong tao duoc mot bai phan tich dung nghia |
| F14 | Thu appeal mo khoa Google | protected | `1.0 -> 0.0` | Unnecessary protection | Mat di su cu the huu ich |
| F15 | Mo rong copy marketing MERN | protected | `0.0 -> 0.0` | Task drift | Yeu cau mo rong bi bien thanh framing san pham |
| F16 | Tom tat bai Palantir | protected | `1.0 -> 0.0` | Grounding loss | Tro nen chung chung khi URL bi yeu di |
| F17 | Tom tat/dich bai analytics | protected | `0.0 -> 0.0` | Context stripping | Cac con so va chi tiet quan trong bi yeu di |
| F18 | Thu xin viec Safran 2 | protected | `0.0 -> 0.0` | Context stripping | Kem tuy bien hon ban goc |
| F19 | Dich thong bao TA | protected | `0.0 -> 0.0` | Task drift | Dinh nghia sai thanh noi dung bai tap |
| F20 | Rewrite loi chao hop tac | protected | `0.0 -> 0.0` | Style flattening | “Better” tro thanh trung tinh |
| F21 | Paraphrase ve phi Telus | protected | `0.0 -> 0.0` | Style flattening | Tro thanh giai thich thay vi paraphrase |
| F22 | Tac vu mo ta link LinkedIn | protected | `0.0 -> 0.0` | Task drift | Tac vu cua nguoi dung bi lam ban boi boi meta prompt generation |
| F23 | Roleplay Omega/Victoria | protected | `0.25 -> 0.0` | Instruction conflict | Roleplay bi thay bang system-style response |
| F24 | Blog telehealth | protected | `0.0 -> 0.0` | Format collapse | Lap prompt thay vi tao bai viet |
| F25 | Viet medical profile | protected | `0.0 -> 0.0` | Task drift | Profile bi doi thanh cover letter |
| F26 | Rephrase doan y khoa | protected | `0.0 -> 0.0` | Context stripping | Giam bot chi tiet quy trinh |

## 10 truong hop structured thuc te da cai thien QUAL

Nhung truong hop duoi day la vi du ma structured delegation da cai thien `QUAL`. Tuy nhien, mot so truong hop dong thoi lam tang `leakage`, nen khong nen xem chung la cac ket qua tot hon mot cach tuyet doi.

| ID | Tom tat truy van | route | thay doi leakage | Vi sao structured tot hon |
|---|---|---|---|---|
| R01 | Thanh cong nghe nghiep cua Caroline Regis | protected | `0.0 -> 0.375` | Ban goc qua do du; structured dua ra cau tra loi suy luan day du hon |
| R02 | Lich su chiec vuong mien Bavaria | protected | `0.0 -> 0.0` | Cau tra loi structured mach lac hon va giau tinh tu su hon |
| R03 | Rut gon doan text ve seller interview | protected | `0.0 -> 0.0` | Ban goc troi sang dua khuyen nghi; structured bam sat hon yeu cau rewrite |
| R04 | Blog OVHcloud | direct | `0.0 -> 1.0` | Output o phia structured bam sat chu de blog thuc te hon |
| R05 | GMB posts | protected | `0.0 -> 0.0` | Tuan thu tot hon rang buoc do dai va keyword |
| R06 | Chien dich social cho Msitu Africa | protected | `0.0 -> 0.0` | Cau truc chien dich trong 1 nam ro rang hon |
| R07 | Van de moi truong cua Ai Cap | protected | `0.0 -> 0.0` | Cau tra loi structured giu noi dung goc trung thanh hon |
| R08 | Chen ten nguoi thang giai dau | protected | `0.0 -> 0.2` | Ban goc do du vi thieu boi canh; structured hoan thanh dung yeu cau van ban |
| R09 | Sua ngu phap email xin phe duyet workshop | protected | `0.0 -> 0.375` | Structured tro thanh mot email doanh nghiep day du hon |
| R10 | Draft email gui Blessing | direct | `0.0 -> 1.0` | Tieu de va mo ta van de ro rang hon ban goc |

## Mau so chung trong cac truong hop duoc cai thien

- Khi ban goc tro nen qua do du hoac noi rang thieu thong tin, structured path doi khi tao ra cau tra loi day du hon.
- Voi cac yeu cau nhu `campaign`, `post`, hoac `email draft`, structured doi khi ap dat duoc cau truc output ro rang hon.
- Tuy nhien, mot so truong hop phuc hoi da cai thien `QUAL` nhung lai tang `leakage`, vi vay khong nen xem day la chien thang ro rang cua privacy pipeline.

## Ket luan

- Phan lon trong 26 truong hop moi that bai den tu `protected / pii_detected` path.
- Cac mau that bai chinh la `task drift`, `context stripping`, va `format collapse`.
- Noi ngan gon, structured delegation pipeline hien tai dang mat nhieu chat luong hon do viet lai chinh tac vu cua nguoi dung so voi loi ich ma no thu duoc tu viec dong khung an toan cho rieng tu.
- 10 truong hop duoc cai thien chu yeu xay ra khi ban goc qua do du hoac cau truc kem.
- Huong cai tien tiem nang nhat la giam viec PII detection bi kich hoat qua muc, va dat ra quy tac manh hon de `structured_task` khong bao gio thay doi hanh dong cot loi cua nguoi dung nhu `translate`, `rewrite`, `summarize`, hoac `draft`.
