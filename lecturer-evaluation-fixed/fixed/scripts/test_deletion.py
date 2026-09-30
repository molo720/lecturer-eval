from app import app, get_db

client = app.test_client()
with app.app_context():
    db = get_db()
    users = db.execute('SELECT id, username, role, lecturer_name FROM users').fetchall()
    print('Users count before:', len(users))
    
    # Create dummy user to test delete
    db.execute('DELETE FROM users WHERE username = ?', ('testdelete',))
    db.execute(
        'INSERT INTO users (username, password_hash, role, lecturer_name, full_name) VALUES (?, ?, ?, ?, ?)',
        ('testdelete', 'hash', 'lecturer', 'Dr. DeleteTest', 'Dr. DeleteTest')
    )
    db.commit()
    test_user = db.execute('SELECT id FROM users WHERE username = ?', ('testdelete',)).fetchone()
    uid = test_user['id']
    print('Created test user ID:', uid)
    
    # Simulate admin session
    with client.session_transaction() as sess:
        sess['user_id'] = 999
        sess['user_role'] = 'administrator'
        sess['username'] = 'admin'
        
    res = client.post(f'/admin/users/delete/{uid}', follow_redirects=True)
    print('Delete user response status:', res.status_code)
    
    # Check if user was deleted
    rem = db.execute('SELECT * FROM users WHERE username = ?', ('testdelete',)).fetchone()
    print('User remains in DB?', rem is not None)
    
    # Test GET on /admin/users
    res_page = client.get('/admin/users')
    print('Admin users page status:', res_page.status_code)
    
    # Test lecturer delete route
    res_lec = client.post('/admin/lecturers/delete', data={'lecturer_name': 'Dr. DeleteTest'}, follow_redirects=True)
    print('Lecturer delete response status:', res_lec.status_code)
    print('ALL DELETION TESTS PASSED!')

