Viral Stories Factory
نظام آلي على GitHub Actions: يكتشف قصصًا حقيقية مثيرة من مصادر محددة → يحلّلها ويختار الأقوى → يكتب مقالًا عربيًا أصليًا (مع فحص حقائق) → يولّد صورة AI جديدة → ينشر على Blogger → يجهّز حزمة فيسبوك (عنوان، منشور، تعليق أول) تشير إلى رابط Blogger الحقيقي → يرسل الحزمة عبر WhatsApp → يحفظ الحالة ويستكمل بعد أي فشل دون تكرار.
المعمارية
```
Sources → Discovery (RSS → Sitemap → Sections) → Dedup (URL / مصدر+عنوان / الحدث عبر Gemini)
→ Triage & Viral selection → Extraction → Content (Gemini JSON Schema) → Fact-check
→ Visual analysis → Image generation (+ validation) → Blogger → REAL URL → Facebook package → WhatsApp → History
```
ملف	مسؤوليته
`main.py`	التنسيق فقط، state machine، recovery
`sources.py` / `extractor.py`	الاكتشاف (robots.txt + rate limit) واستخراج المقال
`history.py`	story_id (SHA-256)، dedup، الحد اليومي، الاستكمال، cache لكل قصة
`gemini_client.py`	كل اتصالات Gemini (JSON Schema + retry + تحقق)
`content.py`	الفرز، كتابة المقال العربي، فحص الحقائق، تنقية HTML
`visual_analyzer.py` / `image_generator.py`	تحليل الصورة المرجعية وتوليد الصورة الجديدة (المزود قابل للاستبدال)
`image_host.py`	نشر الصور في المستودع كرابط عام (يحتاجه Blogger وTwilio)
`blogger.py` / `facebook.py` / `twilio_whatsapp.py`	النشر، حزمة فيسبوك، الإشعار
سياسة الصور
صور الأشخاص، بمن فيهم القاصرون، تُعاد معالجتها مع الحفاظ على الوجوه الظاهرة عند توافر صورة مرجعية. لا يضمن مزود التوليد تطابق الوجه حرفيًا.
صورة Facebook مربعة (1:1)، وتحاول جلب صورتين أو ثلاث من صفحة المصدر، ثم توليد نسخ AI مستوحاة منها وتركيبها في قالب يختاره النموذج: صورة رئيسية مع inset دائري/مربع، لوحتان متجاورتان/متراكبتان، أو triptych.
لا تُرسم عناوين أو شروح أو شعارات أو علامات مائية فوق الصورة. يُطلب الحفاظ على اللافتات الأصلية ذات الصلة عند ظهورها بوضوح، لكن دقة إعادة إنتاج الكتابة ليست مضمونة.
صور المصدر نفسها لا تُستخدم كصورة Facebook النهائية. إذا توفر مصدر واحد فقط، يُنشأ المنظور الثانوي بالاستناد إليه؛ وإذا لم تتوفر صور صالحة، يُنشأ المشهد من سياق القصة. عند تعذر توليد اللوحة كاملة، يستخدم النظام لوحة مربعة احتياطية من صورة المقال.
صورة المقال مستقلة بنسبة 16:9؛ أما Facebook فله صورة مربعة مستقلة عند `FACEBOOK_SEPARATE_IMAGE=true`.
الافتراضي الآن `FLUX.2 dev` مع 25 خطوة وطول ضلع 1536px بدل نموذج Klein السريع ذي 4 خطوات و1024px؛ يتوقع أن يكون أبطأ وقد يستهلك استخدامًا أكبر من مزود الصور.
الإعداد
أنشئ مستودعًا عامًا (لأن الصور تُقدَّم عبر `raw.githubusercontent.com`)، أو ضع `IMAGE_PUBLIC_BASE_URL` لرابط عام آخر (GitHub Pages، CDN...).
أضف Secrets في Settings → Secrets and variables → Actions:
`GEMINI_API_KEY`, `BLOGGER_BLOG_ID`, `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REFRESH_TOKEN`, `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_WHATSAPP_NUMBER`, `YOUR_PERSONAL_NUMBER` (واختياريًا `IMAGE_API_KEY`, `WHATSAPP_CONTENT_SID`).
اختياري: Variables بنفس أسماء الإعدادات (`TARGET_DAILY_STORIES`, `GEMINI_MODEL`, `GEMINI_IMAGE_MODEL`, `FACEBOOK_LAYOUT`...).
Settings → Actions → General → Workflow permissions: Read and write.
Blogger OAuth
في Google Cloud: فعّل Blogger API v3، أنشئ OAuth Client، ثم احصل على refresh token بنطاق `https://www.googleapis.com/auth/blogger` (مثلًا عبر OAuth 2.0 Playground مع استخدام client خاص بك). `BLOGGER_BLOG_ID` تجده في رابط لوحة التحكم.
Twilio WhatsApp
داخل نافذة 24 ساعة (بعد أن ترسل لرقم Sandbox/المرسل رسالة) تُرسل رسائل حرة مع الصورة.
خارجها يلزم قالب معتمد: ضع `WHATSAPP_CONTENT_SID` (متغيران: `{{1}}` العنوان، `{{2}}` الرابط). بدونه تُسجَّل `whatsapp_status=failed` ويُعاد الإرسال لاحقًا دون إعادة النشر.
حدود الوسائط: الصور حتى 5MB، ونوع المحتوى يجب أن يكون صحيحًا؛ يفحص النظام الرابط قبل الإرسال.
التشغيل
محليًا:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env   # املأ القيم ثم: export $(grep -v '^#' .env | xargs)
python -m src.main --dry-run      # آمن: لا نشر، لا واتساب، لا تعديل للتاريخ
python -m src.main --live --stories 1
pytest
```
على GitHub: Actions → Viral Stories Factory → Run workflow (`dry_run=true` للتجربة). التشغيل المجدول (كل 4 ساعات) حقيقي دائمًا. في Dry Run تُحفظ المعاينات في `data/dry_run/` وتُرفع كـ artifact (توليد الصور فيه معطّل ما لم تضبط `DRY_RUN_GENERATE_IMAGES=true`).
الهدف اليومي والجودة
`TARGET_DAILY_STORIES=6` و`MAX_STORIES_PER_RUN=3`. لا يتجاوز النظام الهدف اليومي (حسب `DAY_TIMEZONE`) إلا مع `--override`. إن لم تتوفر قصص قوية يُنشر أقل ويُسجَّل: `available_valid_stories / selected_stories / rejected_stories / reason_for_shortage`.
History والاستكمال
`data/history.json` (حقول المواصفات + حقول تتبع إضافية) و`data/cache/<story_id>.json` (المقال والتحليل والمحتوى والصورة كي لا تُدفع تكلفة Gemini مرتين). الحالات: `discovered → selected → generated → image_generated → blogger_published → facebook_ready → whatsapp_sent → completed` (+ `image_failed`, `blogger_failed`, `whatsapp_failed`, `failed`).
نجح Blogger وفشل واتساب؟ التشغيل التالي يكمل من واتساب فقط.
انتهت مهلة Blogger؟ يُبحث عن المقال (علامة `story_id` مخفية + العنوان) قبل أي إعادة محاولة.
يتوقف التكرار بعد `MAX_STORY_ATTEMPTS` (3) فتصبح الحالة `failed`.
Deduplication (3 طبقات)
`normalized_url` (بدون tracking params، مع canonical) 2. نفس المصدر + عنوان مشابه (≥ 0.82) 3. نفس الحدث عبر Gemini (داخل الدفعة ومقابل آخر 60 قصة منشورة).
المصادر وحدودها
Bored Panda، BuzzFeed، Bright Side، Reuters، AP، Daily Mail، The Sun. الترتيب: RSS ← Sitemap ← صفحات الأقسام. يحترم النظام `robots.txt` وCrawl-delay ويفصل بين الطلبات (`PER_HOST_DELAY_SECONDS`). روابط RSS/Sitemap الافتراضية في `sources.py` لم أتمكن من اختبارها من بيئتي، وقد يحجب بعض المواقع الزواحف أو يغيّر مساراته؛ المصدر الفاشل يُتجاوز ويُسجَّل. عدّل المسارات عبر `SOURCES_OVERRIDE_FILE` (JSON). تأكد من شروط استخدام كل مصدر ومن ترخيص إعادة استخدام المحتوى (خصوصًا Reuters وAP) قبل النشر التجاري.
استكشاف الأخطاء
العَرَض	السبب/الحل
`No candidates` من مصدر	تحقق من روابط RSS/Sitemap أو robots.txt؛ استخدم `SOURCES_OVERRIDE_FILE`
`image_failed`	رفض/فلترة من مزود الصور أو فشل التحقق؛ جرّب `GEMINI_IMAGE_MODEL` آخر أو `IMAGE_REQUIRED=false`
`public URL not reachable`	المستودع غير عام أو تأخر CDN؛ اضبط `IMAGE_PUBLIC_BASE_URL`
`whatsapp_failed` + outside window	أرسل رسالة لرقم Twilio لفتح النافذة أو اضبط `WHATSAPP_CONTENT_SID`
`invalid JSON` متكرر	غيّر `GEMINI_MODEL`؛ الطبقة معزولة في `gemini_client.py`
الأمان
لا تُطبع الأسرار في السجلات (فلتر redaction)، وكلها عبر GitHub Secrets. صلاحية الـworkflow `contents: write` فقط. الملف `.env` ضمن `.gitignore`.
ملاحظة أسماء النماذج
أسماء نماذج Gemini تتغير بسرعة؛ القيم الافتراضية (`gemini-3-flash-preview` للنص، `gemini-2.5-flash-image` للصور) مأخوذة من وثائق Google وقت البناء. تحقق من القائمة الحالية وعدّلها من الإعدادات دون لمس الكود.
