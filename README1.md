## بسمه تعالی

## حل سودوکو به کمک ابزارهای بینایی ماشینی

## درس بینایی ماشین

## دکتر محمدی استاد:آقای

## ۹۸زمستان

## -------------------------------------------------------------------------------------------------

## دانیال کمالی

## 95521396

## علی صفرپوردهکردی

## 95521279

## حل در این پروژه سعی شد به کمک ابزارهای بینایی ماشین با دریافت تصویر سودوکو اعداد درونی آن را خوانده و تصویر جدول

## شده را در خروجی نمایش دهیم.

```
توان استفاده نمود.و برای تحلیل اینکه هر رقم می ip_cameraبرای دریافت تصویر از دو ابزار وبکم لپتاپ و همچنین استفاده از
استفاده گردیده است. template_matching بیانگر چه عددی است نیز از دو راهبرد شبکه های عمیق و
```
```
سی قسمت های مختلف کد و کاربرد هریک میپردازیم. در ادامه به برر
```
model = load_model('m_kheyli_hast.h5')
در این سطر مدل مربوط به یادگیری عمیق بارگذاری گردیده است.در این راستا از یک شبکه یادگیری عمیق استفاده نموده ایم که
لاسی به آن پرداخته شد و اکنون از شرح با جزئیات آن خودداری پیش ازاین در تمارین ک.تعلیم دیده است hoda-datasetبراساس
نماییم.صرفا این شبکه براساس لایه های کانوولوشنی به کمک ابزار کراس پیاده سازی شده است. می
بهبود شبکه:ما متوجه ضعف های شبکه برای تشخیص ارقام خود شدیم در نتیجه عکس های مربوط به تمپلیت ها را چندین مرتبه به
شوند. fitآن ها agumentation عنوان ورودی به شبکه دادیم تا بیشتر یادبگیرد.و سعی شد بر روی این تمپلیت ها و

```
تفاده شده استخراج شده اند .در زیر نمونه این تمپلیت ها را مشاهده مینمایید.تمپلیت های مذکور از خود جدول سودکو اس
```
## یکی از مشکلاتی که در هنگام حل جدول در دنباله ای از تضاویر رویت شد مشکل گیر افتادن در حل جدول بود.همچنین ایراد

## در صورت زیاد شدن مدت حل برنامه را به ادامه کار وادارد و احتمالی جداول نیز باید مدنظر داشت.پس تایمری در نظر گرفته که

## از هنگ کردن برنامه جلوگیری نماید.

templates = []
for i in range( 1 , 10 ):
templates.append(cv2.imread('./{}.png'.format(i), 0 ))


```
قرار میدهد. templatesاین تکه کد تمپلیت های موردنظر ما را از ورودی دریافت مینماید و در لیست
```
video_capture = cv2.VideoCapture( 0 )
در صورتی که بخواهیم از وبکم تصویر دریافت نماییم این قطعه کدکاربرد خواهد داشت.در ادامه به شرح مفصل آن خواهیم پرداخت.

```
def findNextCellToFill(grid, i, j)
def isValid(grid, i, j, e)
def solveSudoku(grid, i= 0 , j= 0 )
این سه تابع مسئول حل سودکو هستند که به صورت یک ماتریس درآمده و درخروجی نیز یک ماتریس خواهیم داشت.
میباشد ولی چون محل بحث ما نیست از آن میگذریم.همچنین back-trackingمدنظر داشته باشید که روش حل جدول به صورت
راهبرد استفاده شده از نظر زمان حل مسیله بسیار سریع بود.
```
```
def process(img):
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ( 2 , 2 ))
greyscale = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2G
RAY)
denoise = cv2.GaussianBlur(greyscale, ( 9 , 9 ), 0 )
thresh = cv2.adaptiveThreshold(denoise, 255 , cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
cv2.THRESH_BINARY, 11 , 2 )
inverted = cv2.bitwise_not(thresh, 0 )
morph = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)
dilated = cv2.dilate(morph, kernel, iterations= 1 )
return dilated
نماید.و لبه ها را تقویت می این تکه کد نشان دهنده یک تابع در راستای پیش پردازش هست نویز تصویر را کاهش داده
نماید.به کمک فیلترهای گوسی نویز نماید.سپس تصویر را تک کاناله می کرنل مورد نیاز ما را تولید می قابل مشاهده است که ابتدا
بع نماید.سپس ترشولد مورد نیاز را بر اساس همسایگی ها تعیین مینماید و سپس تابع بعدی آنرا واروون میکند.تاگیری می
مورفولوژی نیز به عنوان یک لبه یاب عمل میکند.در واقع واروون ساز قبلی برای آماده سازی ورودی این مرحله بود.
```
## در ادامه نیز عملیات افزایش صورت میگیرد تا لبه ها به اندازه کافی بزرگ شوند.

## قسمت بعدی کد که استفاده گردیده است:

```
def get_corners(img):
contours, hire = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SI
MPLE)
contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
if len(contours) > 0 :
```

largest_contour = np.squeeze(contours[ 0 ])
sums = [sum(i) for i in largest_contour]
differences = [i[ 0 ] - i[ 1 ] for i in largest_contour]
top_left = np.argmin(sums)
top_right = np.argmax(differences)
bottom_right = np.argmax(sums)
bottom_left = np.argmin(differences)
corners = [largest_contour[top_left], largest_contour[top_right], largest
_contour[bottom_left],
largest_contour[bottom_right]]
corners = np.float32(corners)
return corners
raise ValueError("NO CONTURE")
گوشه جدول کرده ایم.به این صورت که کانتورهای تصویر را دریافت میکنیم.و در خط دوم ۴در این تکه کد تلاش برای یافتن
ابتدا اولین کانتور یعنی بزرگترین کانتور را ایم.در صورتی که کانتورهایی موجود باشند براساس مساحت کانتورها سورت نموده
گوشه تصویر را تشخیص میدهیم. ۴میکنیم و squeeze اختار آن را انتخاب نموده و س

```
گوشه تصویر را بیابد. ۴هرکانتور با دو مولفه مختصاتی تعیین میشود در نتیجه به کمک جمع دومولفه و تفاضل دو مولفه توانسته
گوشه کانتر را بازمیگرداند. ۴تابع درنهایت
```
def get_block_num_TM(block):
i = 1
values = []
img = np.copy(block)
for template in templates:
res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
indx = unravel_index(res.argmax(), res.shape)
values.append(res[indx])
if res[indx] > 0.9:
return i
i+= 1
if max(values) < 0.6:
return 0
return np.argmax(np.array(values)) + 1
و مقایسه نموده گرداند.مقدار هریک از خانه ها را برمی template_matching براساس در آن عی هست کهاین تکه کد بیانگر تاب
اینجا -ته اگر هیچ از یک از تمپیلت ها به ترشولدیب و اگر تمپلیت معینی میزان بیشتری داشته باشد به عنوان رقم انتخاب میشود ال
نرسد فرض میکنیم خانه خالی بوده است. - ۰.۶

def get_block_num(block):
img = np.copy(block)
img = cv2.adaptiveThreshold(img, 255 , cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THR
ESH_BINARY, 199 , 2 )
img = cv2.bitwise_not(img, 0 )
cv2.imshow('Video', img)
img = img.astype(np.float32)
img = cv2.resize(img, ( 32 , 32 ))
img1 = np.array(img).reshape(( 1 , 32 , 32 , 1 ))


```
x = model.predict(img1)
return np.argmax(x[ 0 ])
این تکه کد برای بازگرداندن عدد براساس پیشبینی شبکه عصبی هست یعنی درصورتی که بخواهیم تابع پیشبینی گر ما براساس شبکه
عصبی کارنماید از این تابع استفاده میکنیم.های
```
def inverse_perspective(img, dst_img, pts):
dst_img = dst_img.copy()
pts_source = np.array([[ 0 , 0 ], [img.shape[ 1 ] - 1 , 0 ], [ 0 , img.shape[ 0 ] - 1 ],
[img.shape[ 1 ] - 1 , img.shape[ 0 ] - 1 ]], dtype='float32'
)
cv2.fillConvexPoly(dst_img, np.ceil(np.array([pts[ 0 ], pts[ 2 ], pts[ 3 ], pts[ 1 ]]
)).astype(int), 0 , 16 )
M = cv2.getPerspectiveTransform(pts_source, pts)
dst = cv2.warpPerspective(img, M, (dst_img.shape[ 1 ], dst_img.shape[ 0 ]))
return dst_img + dst
سیاه نموده تا سطح معین شده را fillconvexpolyگوشه تصویر اصلی برمیگردانیم.تابع نقطه ۴این تابع نقاطی که داریم را به
بتوان قسمت جدید را با شمل اصلی جمع نموده و جایگزین نماییم.در واقع قسمت سودکو حل نشده پاک شده و قسمت حل شده جایگزین
میگردد.

```
def get_ip_camera():
url = 'http://192.168.43.107:8080/shot.jpg'
img_res = requests.get(url)
img_arr = np.array(bytearray(img_res.content), dtype=np.uint8)
input_img = cv2.imdecode(img_arr, - 1 )
return input_img
def get_webcam():
ret,input_img = video_capture.read()
return input_img
یابند.این دو تابع برای دریافت تصویر ورودی هستند که به ترتیب برای کاربا دوربین موبایل و وبکم لپتاپ کاربرد می
```
```
مربوط به تاریخچه آنچه در ماتریس دیده شده میباشد. History
جواب ماتریس هست. sudoku_resualts
شده ماتریس است. wrap نقطه ۴مربوط به wraped
هم جواب هست. answer
```
```
ما از سودوکو های فارسی استفاده کرده ایم برای اینکه بتوانیم فارسی بنویسیم از
```
```
en2fa = {
0 : '۰', 1 : '۱', 2 : '۲', 3 : '۳', 4 : '۴', 5 : '۵', 6 : '۶', 7 : '۷', 8 : '۸', 9 : '۹'
}
fontFile = "b_nazanin.ttf"
font = ImageFont.truetype(fontFile, 60 )
استفاده کرده ایم و دو سطر اخر نیز برای آماده سازی فونت فارسی مورد نظر ما هستند.
```
```
def prespective_transform(points,input_img):
```

```
pts = np.array([[ 0 , 0 ], [maxHeight - 1 , 0 ], [ 0 , maxWidth - 1 ], [maxHeight - 1
, maxWidth - 1 ]], dtype="float32")
M = cv2.getPerspectiveTransform(points, pts)
dst = cv2.warpPerspective(input_img, M, (maxHeight, maxWidth))
return dst
تابع نیز براساس نقاط بدست آمده قسمت مربوطه را به یک جدول مپ میکند تا بررسی کنیم. این
```
```
def matrix_not_in_history(matrix):
if len(history) == 0 :
return True
x = np.abs(np.array([np.sum(matrix- past) for past in history]))
if(np.min(x)> 50 ):
return True
return False
این تابع چک میکند که آیا تابع را در ماتریس دیده ایم یا نه.
```
```
def get_matrix_index(matrix):
for i,past in enumerate(history):
if np.sum(np.array(past)- np.array(matrix))< 50 :
return i
در تکمیل تابع قبلی در صورتی که ماتریس را دیده باشد ایندکس را برمیگرداند.
توجه شود دو تابع بالا در مورد مختصات گوشه ها بحث مینمایند.
```
```
ت ابهامی ایجاد نمایند را مشاهده فرمایید:ادامه کد ترکیبی از توابع بالاست.صرفا قسمت هایی که ممکن اس
```
if np.sum(matrix)> 100 :
شوند چراکه به ص.رت تخمینی حداقل به چنین مقداری نیاز است و ۱۰۰۰در اینجا بررسی کرده ایم که اعداد درون ماتریکس حداقل
همان شکل ورودی را در خروجی م در غیر این صورت احتمالا شکل ماتریس ناقصی را تشخیص داده است.در غیر این صورت ه
نشان میدهد.

```
مرور مینماییم.را ادامه شرح مختصری از کد
قفل کدن برنامه جلوگیری دو خط اول انتخاب بین استفاده از وبکم و آیپی کمرا هست.در هر پروسس یک تایمر فعل مینماییم تا از
دوکو را ترنسفورم نموده و به یک ده و گوشه های جدول را استخراج نماییم.سپس جدول سویش پردازش نموفریم را پ گردد.سپس
لیل مینماییم که چه قسمت تقسیم نموده و هرقسمت را تح ۹*۹مربع تبدیل منماییم.در صورت گذر از شرط فوق الذکر شکل را به
یم و رسشان نگهداری میشود تا بعدا پر گردند.ماتریس را حل مینمای آد باشد.و در صورتی که خانه خالی تشخیص داده شوند عددی می
خروجی را به تاریخچه اضافه اعداد مربوط به خانه های خالی را پر مینماییم.سپس یک پرسپکتیو معکوس اعمال مینماییم.همچنین
و در صورتی که جدول تشخیص داده نشود م.ل مینمایی هم ابتدا آنرا ح نباشد ماتریسهای حل شده میکنیم.در صورتی که در تاریخچه
هیج محاسباتی خود آنرا نشان میدهیم.بدون انجام
```
```
چالش های موجود: مهمترین
```
```
تشخیص بخشی از جدول به عنوان کانتور •
کیفیت دوربین وبکم و حتی فیلم برداری با توجه به لرزش دست و ... که منتهی به نویز میشد •
قرار میدهد. لکرد شبکه عصبی را تحت شعاع منویز موجود به ویژه ع •
```

